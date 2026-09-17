"""OpenAI-compatible LLM client for ChemEagle.

Everything the agents need from a chat model goes through here, so ChemEagle
runs against any OpenAI-compatible endpoint (OpenAI itself, OpenRouter, an Azure
APIM gateway, vLLM, Ollama, ...) without touching the agent code.

Configuration is environment variables only:

    export API_KEY=your-api-key            # required
    export BASE_URL=https://api.openai.com/v1   # optional, this is the default
    export MODEL=gemini-3.7-flash          # optional, this is the default

``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` are accepted as aliases, and every
value can also be passed in code (``llm_client.configure(...)``) or per call.

Per-model parameter rules, encoded in :func:`model_kwargs`, because endpoints
reject what a model does not support:

    gpt-5.6 family        temperature only with reasoning_effort "none" (the default); max_completion_tokens
    gpt-5-mini / -nano    temperature NOT accepted (only the default 1); max_completion_tokens
    o1 / o3 / o4 family   temperature NOT accepted; max_completion_tokens
    gemini-*              temperature OK; reasoning_effort "none" rejected; max_tokens
    everything else       temperature OK; max_tokens; SAMPLING_PARAMS (JSON) overrides sampling

Transport: requests to the gpt / o-series / gemini families are sent as streams and reassembled
into an ordinary ``ChatCompletion`` (``STREAM_REQUESTS=0`` turns this off, ``=1`` forces it for every
model), because API gateways drop non-streaming connections that have not answered within about
60 seconds, which long reasoning calls exceed. Local OpenAI-compatible servers are left non-streaming.

Nested tool helpers (``get_reaction`` etc.) are not given the model
explicitly; they fall back to the process-wide defaults set with
:func:`configure`, which ``ChemEagle`` calls on entry.
"""
import json
import os
import re
import time
from typing import Optional

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gemini-3.7-flash"

_defaults = {"model_name": None, "api_key": None, "base_url": None, "reasoning_effort": None}


def configure(model_name: Optional[str] = None, api_key: Optional[str] = None,
              base_url: Optional[str] = None, reasoning_effort: Optional[str] = None) -> dict:
    """Set process-wide defaults used by every agent that is not given the value
    explicitly. Passing ``None`` leaves that field unchanged."""
    for key, value in (("model_name", model_name), ("api_key", api_key),
                       ("base_url", base_url), ("reasoning_effort", reasoning_effort)):
        if value is not None:
            _defaults[key] = value
    return dict(_defaults)


def resolve_model(model_name: Optional[str] = None) -> str:
    return model_name or _defaults["model_name"] or os.getenv("MODEL") or DEFAULT_MODEL


def resolve_key(api_key: Optional[str] = None) -> str:
    key = api_key or _defaults["api_key"] or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "API key missing: set the API_KEY environment variable (or OPENAI_API_KEY), "
            "pass api_key=..., or call llm_client.configure(api_key=...)."
        )
    return key


def resolve_base_url(base_url: Optional[str] = None) -> str:
    return (base_url or _defaults["base_url"] or os.getenv("BASE_URL")
            or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def get_client(api_key: Optional[str] = None, base_url: Optional[str] = None,
               timeout: float = 600.0, max_retries: int = 2) -> OpenAI:
    """OpenAI SDK client pointed at ``BASE_URL``.

    The key also goes out as an ``api-key`` header, which Azure-style API
    gateways require and plain OpenAI-compatible servers ignore."""
    key = resolve_key(api_key)
    client = OpenAI(base_url=resolve_base_url(base_url), api_key=key,
                    default_headers={"api-key": key}, timeout=timeout, max_retries=max_retries)
    completions = client.chat.completions
    completions.create = streaming_create(completions.create)
    return client


def _stream_wanted(model_name: Optional[str]) -> bool:
    mode = (os.getenv("STREAM_REQUESTS") or "auto").strip().lower()
    if mode in ("1", "true", "yes", "on", "always"):
        return True
    if mode in ("0", "false", "no", "off", "never"):
        return False
    return model_family(resolve_model(model_name)) != "other"


GATEWAY_STATUS = (502, 503, 504)


def retry_gateway(call, attempts=4, base_delay=3.0):
    """Retry a request the gateway itself refused (a 502 Bad Gateway page from the Azure front end, or a 503 /
    504 while the upstream deployment restarts): the request never reached the model and the same one usually
    goes through seconds later. Everything else, including any 4xx and any error raised after the model
    answered, is re-raised at once. Callers that already retry (``retry_api_call``, ``final_json_call``) keep
    working: this only makes the client itself survive a blip, which the planner call had no cover for."""
    from openai import APIStatusError
    for attempt in range(attempts):
        try:
            return call()
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            if status not in GATEWAY_STATUS or attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[llm_client] gateway {status}; retrying in {delay:.0f}s ({attempt + 2}/{attempts})", flush=True)
            time.sleep(delay)


def streaming_create(original):
    """Wrap ``client.chat.completions.create`` so a non-streaming request goes out as a stream and comes back
    as the same ``ChatCompletion`` object the caller would have received; only the transport changes.

    Streaming keeps a long generation alive through gateways that cut connections idle for about 60 seconds.
    It cannot help while a reasoning model is still thinking and has sent nothing: that cut-off arrives as a
    connection error before any chunk, and it is not retried here because the same request would be cut
    again (the SDK's own connection retries still apply). A stream that breaks after data started flowing
    is retried once from the start. Requests that already set ``stream=True``, and models for which
    :func:`_stream_wanted` is false, pass through unchanged."""
    def create(*args, **kwargs):
        if kwargs.get("stream") or not _stream_wanted(kwargs.get("model")):
            return retry_gateway(lambda: original(*args, **kwargs))
        kw = dict(kwargs, stream=True)
        options = dict(kw.get("stream_options") or {})
        options.setdefault("include_usage", True)
        kw["stream_options"] = options

        def run(received):
            return _collect_completion(_counted(original(*args, **kw), received))
        return retry_gateway(lambda: retry_broken_stream(run))

    return create


# Text of errors meaning a stream broke after it started, either between us and the gateway or between the
# gateway's proxy and the upstream provider (the proxy reports the latter as an API error event in the stream).
_BROKEN_STREAM_MARKERS = ("Response payload is not completed", "APIConnectionError", "Connection error",
                          "incomplete chunked read", "peer closed connection", "Server disconnected")


def _counted(stream, received):
    for chunk in stream:
        received[0] += 1
        yield chunk


def retry_broken_stream(run, attempts: int = 2):
    """Call ``run(received)``, which opens and reads one stream while counting chunks in ``received[0]``,
    and start it again when the stream broke after at least one chunk arrived.

    Nothing is retried when no chunk arrived (an idle cut-off while the model is still reasoning repeats on
    retry), nor for an API error whose text does not say the connection broke (a 400 for a bad parameter)."""
    import httpx
    from openai import APIConnectionError, APIError
    for attempt in range(attempts):
        received = [0]
        try:
            return run(received)
        except (httpx.TransportError, APIError) as exc:
            broken = (isinstance(exc, (httpx.TransportError, APIConnectionError))
                      or any(marker in str(exc) for marker in _BROKEN_STREAM_MARKERS))
            if not broken or received[0] == 0 or attempt == attempts - 1:
                raise
            print(f"[llm_client] stream broke off after {received[0]} chunks ({type(exc).__name__}: "
                  f"{str(exc)[:120]}); retrying the request ({attempt + 2}/{attempts})", flush=True)
            time.sleep(3)

def _collect_completion(stream):
    """Reassemble a chat-completion stream into a ``ChatCompletion``, built the way the SDK builds parsed
    responses, so callers can read it, dump it, or append its message to the next request."""
    from openai.types.chat import ChatCompletion
    meta, content, refusal, calls, finish, usage = {}, {}, {}, {}, {}, None
    for chunk in stream:
        if chunk.id and not meta.get("id"):
            meta["id"] = chunk.id
        meta["created"] = meta.get("created") or chunk.created
        meta["model"] = chunk.model or meta.get("model")
        if getattr(chunk, "system_fingerprint", None):
            meta["system_fingerprint"] = chunk.system_fingerprint
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        for choice in chunk.choices or []:
            i = choice.index or 0
            delta = choice.delta
            if delta is not None:
                if delta.content:
                    content.setdefault(i, []).append(delta.content)
                if getattr(delta, "refusal", None):
                    refusal.setdefault(i, []).append(delta.refusal)
                for tc in delta.tool_calls or []:
                    slots = calls.setdefault(i, {})
                    key = tc.index
                    if key is None:
                        key = next((k for k, v in slots.items() if tc.id and v["id"] == tc.id), len(slots))
                    slot = slots.setdefault(key, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function is not None:
                        slot["function"]["name"] += tc.function.name or ""
                        slot["function"]["arguments"] += tc.function.arguments or ""
            if choice.finish_reason:
                finish[i] = choice.finish_reason
    choices = []
    for i in sorted(set(content) | set(refusal) | set(calls) | set(finish)) or [0]:
        tool_calls = [slot for _, slot in sorted(calls.get(i, {}).items())]
        for n, slot in enumerate(tool_calls):
            slot["id"] = slot["id"] or f"call_{i}_{n}"
        message = {"role": "assistant", "content": "".join(content[i]) if i in content else None}
        if i in refusal:
            message["refusal"] = "".join(refusal[i])
        if tool_calls:
            message["tool_calls"] = tool_calls
        choices.append({"index": i, "logprobs": None, "message": message,
                        "finish_reason": finish.get(i) or ("tool_calls" if tool_calls else "stop")})
    data = {"id": meta.get("id") or "", "object": "chat.completion", "created": meta.get("created") or int(time.time()),
            "model": meta.get("model") or "", "choices": choices,
            "usage": usage.model_dump() if usage is not None else None}
    if meta.get("system_fingerprint"):
        data["system_fingerprint"] = meta["system_fingerprint"]
    return ChatCompletion.construct(**data)


def model_family(model_name: str) -> str:
    name = (model_name or "").lower()
    name = name.split("/", 1)[1] if "/" in name else name
    if name.startswith("gpt-5.6"):
        return "gpt-5.6"
    if name.startswith(("gpt-5-mini", "gpt-5-nano", "gpt-5-")):
        return "gpt-5-small"
    if name.startswith(("o1", "o3", "o4")):
        return "o-series"
    if name.startswith("gemini"):
        return "gemini"
    return "other"


_OPENAI_SAMPLING_KEYS = ("temperature", "top_p", "presence_penalty", "frequency_penalty", "seed", "max_tokens")


def sampling_params() -> dict:
    """Sampling overrides for open models behind an OpenAI-compatible server (vLLM, SGLang, ...), read from
    the ``SAMPLING_PARAMS`` environment variable as a JSON object, for example the Qwen3-VL recommendation
    ``{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0, "presence_penalty": 1.5}``.
    Standard OpenAI fields are sent as such; server-specific ones (top_k, min_p, repetition_penalty, ...)
    go in ``extra_body``. Only models outside the gpt / o-series / gemini families use it."""
    raw = os.getenv("SAMPLING_PARAMS")
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("SAMPLING_PARAMS must be a JSON object")
    return value


def model_kwargs(model_name: Optional[str] = None, *, temperature: Optional[float] = 0,
                 reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None) -> dict:
    """Chat-completion keyword arguments this model accepts.

    ``temperature`` is dropped for families that reject it. ``reasoning_effort``
    defaults per family (``none`` for gpt-5.6, the API default elsewhere) and can
    be overridden by the argument or the ``REASONING_EFFORT`` environment
    variable; Gemini never receives ``none``, which it rejects."""
    family = model_family(resolve_model(model_name))
    effort = reasoning_effort or _defaults["reasoning_effort"] or os.getenv("REASONING_EFFORT")
    kwargs = {}
    if family == "gpt-5.6":
        effort = effort or "none"
        # temperature is accepted only while reasoning is off; with reasoning on the API allows only the default
        if temperature is not None and effort == "none":
            kwargs["temperature"] = temperature
        kwargs["reasoning_effort"] = effort
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
    elif family in ("gpt-5-small", "o-series"):
        if effort and effort != "none":
            kwargs["reasoning_effort"] = effort
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
    elif family == "gemini":
        if temperature is not None:
            kwargs["temperature"] = temperature
        if effort and effort != "none":
            kwargs["reasoning_effort"] = effort
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        for key, value in sampling_params().items():
            if key in _OPENAI_SAMPLING_KEYS:
                kwargs[key] = value
            else:
                kwargs.setdefault("extra_body", {})[key] = value
    return kwargs


def _stream_to_message(chunks):
    """Assemble a streamed chat completion into the small message shape final_json_call reads."""
    from types import SimpleNamespace
    content, calls, finish = [], {}, None
    for chunk in chunks:
        for choice in getattr(chunk, "choices", None) or []:
            delta = choice.delta
            if delta is not None and delta.content:
                content.append(delta.content)
            for tc in (delta.tool_calls if delta is not None and delta.tool_calls else []):
                slot = calls.setdefault(tc.index, {"id": tc.id or "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function is not None:
                    slot["name"] += tc.function.name or ""
                    slot["arguments"] += tc.function.arguments or ""
            if choice.finish_reason:
                finish = choice.finish_reason
    tcs = [SimpleNamespace(id=v["id"], function=SimpleNamespace(name=v["name"], arguments=v["arguments"])) for _, v in sorted(calls.items())]
    as_dict = {"role": "assistant", "content": "".join(content) or None}
    if tcs:
        as_dict["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in tcs]
    message = SimpleNamespace(role="assistant", content=as_dict["content"], tool_calls=tcs or None, as_dict=as_dict)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=None)


_LENIENT_BS = re.compile(r'\\\\|\\(?!["\\/bfnrtu])')


def loads_lenient(text):
    """json.loads with two repairs for model output: strip ``` fences, and double any lone backslash
    that does not start a valid JSON escape (models write SMILES E/Z bonds such as C=C(\\[2*]) raw).
    An already escaped pair `\\\\` is kept as a unit, so mixed escaped/raw content is repaired too."""
    if text is None:
        raise json.JSONDecodeError("empty content", "", 0)
    t = str(text).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        t2 = _LENIENT_BS.sub(lambda m: "\\\\" if len(m.group(0)) == 1 else m.group(0), t)
        return json.loads(t2)


def dump_unparsable(text, tag="model_output"):
    """Save an unparsable model response next to the log so it can be diagnosed offline."""
    try:
        d = os.path.join(os.environ.get("CHEMEAGLE_DEBUG_DIR", os.getcwd()), "unparsable_json")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{tag}_{int(time.time())}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text))
        return path
    except Exception:
        return None


def final_json_call(client: OpenAI, model_name: str, messages: list, mk: dict, *,
                    tool_map: Optional[dict] = None, tool_arg=None, retry=None, max_rounds: int = 4,
                    stream: bool = False):
    """Final "answer as one JSON object" call of an agent, made robust to models
    that do not finish in one turn.

    The agents ask for tools in round one and expect the JSON answer in round
    two. Some models (Gemini in particular) call tools one at a time or answer a
    JSON-mode request with another tool call / empty content, which the original
    code turned into ``json.loads(None)``. This helper:

      * executes any further tool calls through ``tool_map`` (name -> callable
        taking ``tool_arg``) and appends the results, then asks again;
      * on empty content without tool calls, appends a one-line reminder and
        asks again;
      * gives up after ``max_rounds`` and returns the last response, so the
        caller's own error handling still applies.

    ``retry`` is the caller's ``retry_api_call`` (503/overload backoff) or None.
    ``stream=True`` streams the reply and reassembles it: a gateway tends to drop
    idle non-streaming connections on long generations (the final synthesis of a
    big scheme), streaming keeps them alive.
    Returns ``(response, messages)`` where ``messages`` includes any extra turns.
    """
    import json

    def create(**kw):
        if stream:
            kw = dict(kw, stream=True)

            def run(received):
                if retry is None:
                    chunks = client.chat.completions.create(**kw)
                else:
                    chunks = retry(client.chat.completions.create, max_retries=6, base_delay=3, backoff_factor=2, **kw)
                return _stream_to_message(_counted(chunks, received))
            return retry_broken_stream(run)
        if retry is None:
            return client.chat.completions.create(**kw)
        # six retries (3 s ... 96 s, about three minutes in all): a gateway 502 window of ~90 s was seen on 2026-09-17
        return retry(client.chat.completions.create, max_retries=6, base_delay=3, backoff_factor=2, **kw)

    messages = list(messages)
    response = None
    for _round in range(max_rounds):
        response = create(model=model_name, messages=messages, response_format={"type": "json_object"}, **mk)
        message = response.choices[0].message
        content = message.content
        if content and content.strip():
            return response, messages
        tool_calls = message.tool_calls or []
        if tool_calls:
            messages.append(getattr(message, 'as_dict', message))
            for call in tool_calls:
                name = call.function.name
                fn = (tool_map or {}).get(name)
                if fn is None:
                    result = {"error": f"tool {name} is not available here; its output is already in the conversation"}
                else:
                    result = fn(tool_arg)
                messages.append({"role": "tool", "name": name, "tool_call_id": call.id,
                                 "content": json.dumps({"image_path": tool_arg, name: result}, ensure_ascii=False)})
            print(f"[llm_client] {model_name}: executed {len(tool_calls)} extra tool call(s) "
                  f"({', '.join(c.function.name for c in tool_calls)}) in JSON round {_round + 1}", flush=True)
        else:
            print(f"[llm_client] {model_name}: empty content in JSON round {_round + 1}, asking again", flush=True)
            messages.append({"role": "user", "content": "All tool results are above. Return the final answer now as a single JSON object."})
    return response, messages
