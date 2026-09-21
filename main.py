import sys
import torch
import json
from chemietoolkit import ChemIEToolkit,utils
import cv2
from openai import AzureOpenAI, OpenAI
import numpy as np
from PIL import Image
import os
from rxnim import RxnIM
import base64
from typing import Optional, Dict, Any
import get_molecular_agent
from get_molecular_agent import process_reaction_image_with_multiple_products_and_text_correctR, process_reaction_image_with_multiple_products_and_text_correctmultiR_azure
from get_reaction_agent import get_reaction_withatoms_correctR_azure, get_reaction_con_azure
from get_R_group_sub_agent import process_reaction_image_with_table_R_group_azure, process_reaction_image_with_product_variant_R_group_azure, get_full_reaction_template_azure, get_multi_molecular_full_azure
from get_observer import action_observer_agent_azure, plan_observer_agent_azure
from get_text_agent import text_extraction_agent_azure
from get_reaction_agent import get_reaction_con
from get_R_group_sub_agent import (process_reaction_image_with_product_variant_R_group,
                                   process_reaction_image_with_table_R_group,
                                   get_full_reaction_template, get_multi_molecular_full,
                                   peek_cached_multi_molecular, drawn_condition_structures)
from get_observer import action_observer_agent, plan_observer_agent
from get_text_agent import text_extraction_agent
import llm_client as llm
import traceback
from chemietoolkit.helper import (_clean_agent_name, _parse_planner_output, _resolve_ordered_agents,
                                 fallback_validate_and_fix_smiles_in_dict, fallback_resolve_condition_smiles_in_data,
                                 fallback_resolve_reactant_product_smiles_in_data, post_verification,
                                 rgroup_fallback_agent, set_network_enabled, network_enabled)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChemIEToolkit(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
ckpt_path = "./rxn.ckpt"
model1 = RxnIM(ckpt_path, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

def _normalize_agent_args(raw_args: Optional[dict], image_path: str) -> dict:
    if not isinstance(raw_args, dict):
        return {"image_path": image_path}
    normalized = dict(raw_args)
    placeholder_values = {"[img]", "<img>", "[image]", "<image>", "<<<IMAGE>>>", "IMAGE_PATH", "image.png","image_path"}
    if normalized.get("image_path") in placeholder_values or normalized.get("image_path") is None:
        normalized["image_path"] = image_path
    return normalized



############################### ChemEagle (any OpenAI-compatible endpoint)
def _coerce_final_output(obj):
    """The final synthesis reply as {'reactions': [...]} whatever shape the model chose:
    a bare list of reactions, a dict keyed differently, or a single reaction dict."""
    if isinstance(obj, list):
        return {'reactions': obj}
    if isinstance(obj, dict):
        if 'reactions' in obj:
            return obj
        for k, v in obj.items():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v) and any('reactants' in x or 'products' in x for x in v):
                out = dict(obj)
                out['reactions'] = out.pop(k)
                return out
        if 'reactants' in obj or 'products' in obj:
            return {'reactions': [obj]}
    return obj


def ChemEagle(
    image_path: str,
    *,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
    use_network: Optional[bool] = None,
    artifacts: Optional[dict] = None,
) -> dict:
    """ChemEagle over any OpenAI-compatible endpoint (replaces the deprecated
    Azure OpenAI route of ``ChemEagle_azure``). Every LLM call goes through
    ``llm_client``, so any OpenAI-compatible endpoint works: a hosted API, a
    gateway, or a local server (vLLM, Ollama).

    Args:
        image_path: path to the reaction image.
        model_name: model id, e.g. gemini-3.7-flash, gpt-5.6-terra, gpt-5-mini
            (default: MODEL env, else gemini-3.7-flash).
        base_url: API base URL (default: BASE_URL env, else https://api.openai.com/v1).
        api_key: API key (default: API_KEY env).
        use_plan_observer / use_action_observer: as in ChemEagle.
        use_network: whether a condition or reagent named in words may be looked up
            over the network (PubChem, OPSIN, the OCSR resolver). True asks the
            services, False leaves the alias map, the cached names and the local
            OPSIN jar to answer alone, which costs no waiting; None (default) keeps
            whatever the CHEMEAGLE_NETWORK / CHEMEAGLE_OFFLINE environment says,
            and the network is on when neither is set. Structures the figure draws
            are read by the recogniser and never depend on this.
        artifacts: optional dict that receives the planner output, the executed agents
            with their results, failures and the raw final reply, for evaluation.
    """

    if use_network is not None:
        set_network_enabled(use_network)
    if not network_enabled():
        print("[network] off: names resolve from the alias map, the cache and local OPSIN only")

    model_name = llm.resolve_model(model_name)

    base_url = llm.resolve_base_url(base_url)
    api_key = llm.resolve_key(api_key)
    _mk = llm.model_kwargs(model_name)
    # Nested tool helpers (get_reaction, get_cached_*, ...) take no
    # model argument; they read the process-wide defaults set here.
    llm.configure(model_name=model_name, api_key=api_key, base_url=base_url)

    client = llm.get_client(api_key=api_key, base_url=base_url)

    def encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    base64_image = encode_image(image_path)

    with open('./prompt/prompt_final_simple_version.txt', 'r', encoding='utf-8') as prompt_file:
    #with open('./prompt/prompt_data.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    with open('./prompt/prompt_plan.txt', 'r', encoding='utf-8') as prompt_file:
        planner_user_message = prompt_file.read()

    planner_response = client.chat.completions.create(
        model=model_name,
        **_mk,
        messages=[
            {'role': 'system', 'content': "You are a chemical image understanding and extraction planning expert. After checking the image, your ONLY task is to SELECT and CALL the most appropriate agents from the list below to best fit the data extraction of the image."},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': planner_user_message},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
                ]
            }
        ]
    )
    
    planner_output = (planner_response.choices[0].message.content or "").strip()
    print(f"[dispatch] Planner output: {planner_output}")
    
    agent_list = _parse_planner_output(planner_output)
    if artifacts is not None:
        artifacts.update(planner_output=planner_output, agent_list=list(agent_list))
    print(f"[dispatch] Parsed agents: {agent_list}")
    
    if use_plan_observer:
        observer_output = plan_observer_agent(image_path, agent_list, model_name=model_name, base_url=base_url, api_key=api_key)
        reviewed = observer_output.get("list_of_agents", agent_list)
        reason = observer_output.get("reason", "")
        if isinstance(reviewed, list) and reviewed:
            new_agents = []
            for item in reviewed:
                if isinstance(item, str):
                    new_agents.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("tool_name") or ""
                    if name:
                        new_agents.append(name)
            if new_agents:
                agent_list = new_agents
                print(f"[dispatch] Plan observer revised agents: {agent_list}")
                if reason:
                    print(f"[dispatch] Plan observer reason: {reason}")
    
    # Resolve the planner's ordered agent list into an executable agent sequence
    # (order-preserving dedup + composite-agent mutual exclusion; see helper).
    ordered_agents, has_text_extraction = _resolve_ordered_agents(agent_list)
    print(f"[dispatch] Ordered agents (planner order): {ordered_agents}")

    AGENT_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group,
        'get_full_reaction_template': get_full_reaction_template,
        'get_multi_molecular_full': get_multi_molecular_full,
        'get_reaction_con': get_reaction_con,
        'text_extraction_agent': text_extraction_agent
    }

    NEW_AGENTS_ACCEPT_MODEL = (
        "process_reaction_image_with_product_variant_R_group",
        "process_reaction_image_with_table_R_group",
        "get_reaction_con",
    )

    def _new_agent_args(agent_name: str) -> dict:
        args = {"image_path": image_path}
        if agent_name in NEW_AGENTS_ACCEPT_MODEL:
            args["model_name"] = model_name
            args["base_url"] = base_url
            args["api_key"] = api_key
        return args

    execution_logs = []
    results = []
    main_area_result = None
    observer_notes = []
    failed_agents = []

    def _observe_and_retry(observed_name, observed_result, rerun):
        """Per-agent Action Observer check; at most one re-execution on redo.
        The observer's diagnosis is collected and forwarded to the final
        synthesis step."""
        check = action_observer_agent(
            image_path, [{"name": observed_name, "result": observed_result}],
            model_name=model_name, base_url=base_url, api_key=api_key)
        if check.get("redo"):
            reason = check.get("reason", "")
            observer_notes.append(f"{observed_name}: {reason}" if reason else observed_name)
            print(f"[dispatch] Action observer requested redo for {observed_name}: {reason}")
            observed_result = rerun()
        return observed_result

    for idx, agent_name in enumerate(ordered_agents):
        print(f"[dispatch] Executing agent {idx + 1}/{len(ordered_agents)}: {agent_name}")
        try:
            agent_result = AGENT_MAP[agent_name](**_new_agent_args(agent_name))
            if use_action_observer:
                agent_result = _observe_and_retry(
                    agent_name, agent_result,
                    lambda: AGENT_MAP[agent_name](**_new_agent_args(agent_name)))
        except Exception as exc:
            # Keep the agents that already succeeded: one failure costs its own
            # contribution, not the whole extraction. The synthesis step is told
            # which agent is missing via observer_reason.
            failed_agents.append(f"{agent_name}: {type(exc).__name__}: {exc}")
            observer_notes.append(f"{agent_name} failed and was skipped ({type(exc).__name__})")
            print(f"[dispatch] Agent {agent_name} failed, continuing without it: {exc!r}")
            traceback.print_exc()
            continue
        if main_area_result is None:
            main_area_result = agent_result
        execution_logs.append({
            "id": f"agent_call_{idx}",
            "name": agent_name,
            "arguments": {"image_path": image_path},
            "result": agent_result,
        })
        results.append({
            'role': 'tool',
            'name': agent_name,
            'content': json.dumps({
                'image_path': image_path,
                agent_name: agent_result,
            }),
            'tool_call_id': f"agent_call_{idx}",
        })

    # The planner sometimes routes a figure that draws an R-group template to the plain template agents, and
    # then every row comes back as a copy of that template. When the molecular agent's own output shows a
    # wildcard, run the R-group agent the plan is missing and let the synthesis use both results.
    if results:
        mol_cached = peek_cached_multi_molecular(image_path) or []
        mol_smiles = [b.get('smiles') for item in mol_cached for b in (item.get('bboxes') or []) if isinstance(b, dict)]
        fallback_name, fallback_reason = rgroup_fallback_agent(ordered_agents, mol_smiles)
        if fallback_name:
            print(f"[dispatch] no R-group agent was planned but {fallback_reason}; running {fallback_name}")
            try:
                fallback_result = AGENT_MAP[fallback_name](**_new_agent_args(fallback_name))
            except Exception as exc:
                failed_agents.append(f"{fallback_name} (fallback): {type(exc).__name__}: {exc}")
                print(f"[dispatch] fallback agent {fallback_name} failed, continuing without it: {exc!r}")
                traceback.print_exc()
            else:
                idx = len(execution_logs)
                execution_logs.append({"id": f"agent_call_{idx}", "name": fallback_name,
                                       "arguments": {"image_path": image_path}, "result": fallback_result})
                results.append({'role': 'tool', 'name': fallback_name,
                                'content': json.dumps({'image_path': image_path, fallback_name: fallback_result}),
                                'tool_call_id': f"agent_call_{idx}"})
                observer_notes.append(f"{fallback_name} was run although the plan omitted it: {fallback_reason}")

    print(f'[dispatch] results: {results}')

    if artifacts is not None:
        artifacts.update(ordered_agents=list(ordered_agents), execution_logs=execution_logs, failed_agents=list(failed_agents))
    if not results:
        raise RuntimeError(
            "All planned agents failed, nothing left to synthesise: "
            + " | ".join(failed_agents)
        )

    observer_reason = "; ".join(observer_notes)

    text_extraction_result = None
    if has_text_extraction:
        print(f"[dispatch] Executing text_extraction_agent with graphical_input")
        try:
            text_extraction_result = text_extraction_agent(
                image_path=image_path,
                graphical_input=main_area_result,
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
            )
            if use_action_observer and text_extraction_result is not None:
                text_extraction_result = _observe_and_retry(
                    "text_extraction_agent", text_extraction_result,
                    lambda: text_extraction_agent(
                        image_path=image_path, graphical_input=main_area_result,
                        model_name=model_name, base_url=base_url, api_key=api_key))
        except Exception as exc:

            failed_agents.append(f"text_extraction_agent: {type(exc).__name__}: {exc}")
            observer_notes.append(f"text_extraction_agent failed and was skipped ({type(exc).__name__})")
            print(f"[dispatch] Agent text_extraction_agent failed, continuing without it: {exc!r}")
            traceback.print_exc()
            text_extraction_result = None
        observer_reason = "; ".join(observer_notes)


    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": res['tool_call_id'],
                "type": "function",
                "function": {
                    "name": log["name"],
                    "arguments": json.dumps({"image_path": image_path}),
                },
            }
            for log, res in zip(execution_logs, results)
        ],
    }

    messages_list = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}' }}
            ],
        },
        assistant_message,
        *results,
    ]

    if text_extraction_result is not None:
        if isinstance(text_extraction_result, dict) and "annotated_text" in text_extraction_result:
            _text_extraction_for_msg = {"annotated_text": text_extraction_result["annotated_text"]}
        else:
            _text_extraction_for_msg = text_extraction_result
        messages_list.append({
        "role": "user",
        "content": (
            "Additionally, the text_extraction_agent has produced the following "
            "JSON for the prose portion of the same image."
            "```json\n"
            + json.dumps(_text_extraction_for_msg, ensure_ascii=False, indent=2)
            + "\n```"
        ),
    })
    
    if observer_reason:
        messages_list.append({
            'role': 'user',
            'content': f"Note: the previous execution had potential errors: {observer_reason}. Please review the results carefully.",
        })

    response, _ = llm.final_json_call(
        client, model_name, messages_list, _mk,
        tool_map=None, tool_arg=image_path, retry=None, stream=True)

    try:
        gpt_output = _coerce_final_output(llm.loads_lenient(response.choices[0].message.content))   # tolerates raw backslashes (E/Z SMILES) and fences
    except json.JSONDecodeError as _e:
        print(f"ERROR [agent]: final synthesis JSON unparsable ({_e}); raw content saved to "
              f"{llm.dump_unparsable(response.choices[0].message.content, 'final_synthesis')}")
        raise   # tolerates raw backslashes (E/Z SMILES) and fences
    gpt_output = post_verification(gpt_output,
                                   label_structures=get_molecular_agent.label_structures(image_path),
                                   drawn_conditions=drawn_condition_structures(image_path))

    if text_extraction_result is not None:
        if isinstance(text_extraction_result, dict) and "annotated_text" in text_extraction_result:
            gpt_output["text_extraction"] = [{"annotated_text": text_extraction_result["annotated_text"]}]
        else:
            gpt_output["text_extraction"] = [text_extraction_result]
    
    if artifacts is not None:
        artifacts.update(text_extraction_result=text_extraction_result, observer_notes=list(observer_notes),
                         failed_agents=list(failed_agents), final_raw=response.choices[0].message.content)
    print(gpt_output)
    return gpt_output




def ChemEagle_azure(
    image_path: str,
    *,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
) -> dict:
    """
    Given a chemical reaction image path, extract reaction information
    using GPT models and specialized agents, and return structured reaction data.
    Supports plan observer and action observer. Default set to False to save token and time.

    Args:
        image_path (str): Path to the image file.
        use_plan_observer (bool): Whether to use plan observer to review the agent call plan.
        use_action_observer (bool): Whether to use action observer to check execution results.

    Returns:
        dict: Structured reaction data including reactants, products, and reaction template.
    """
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)


    agent_specs = [
        {
        'type': 'function',
        'function': {
            'name': 'process_reaction_image_with_product_variant_R_group',
            'description': 'get the reaction data of the reaction diagram and get SMILES strings of every detailed reaction in reaction diagram and the set of product variants, and the original molecular list.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
            {
        'type': 'function',
        'function': {
            'name': 'process_reaction_image_with_table_R_group',
            'description': 'get the reaction data of the reaction diagram and get SMILES strings of every detailed reaction in reaction diagram and the R-group table',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
            {
        'type': 'function',
        'function': {
            'name': 'get_full_reaction_template',
            'description': 'After you carefully check the image, if this is a reaction image that contains only a text-based table and does not involve any R-group replacement, or this is a reaction image does not contain any tables or sets of product variants, then just call this simplified agent.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
            {
        'type': 'function',
        'function': {
            'name': 'get_multi_molecular_full',
            'description': 'After you carefully check the image, if this is a single molecule image or a multiple molecules image, then need to call this molecular recognition agent.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
        {
        'type': 'function',
        'function': {
            'name': 'get_reaction_con',
            'description': 'Extract and normalize the reaction conditions (catalysts, reagents, solvent, temperature, time, atmosphere, etc.) from the graphic. Call this condition interpretation agent when the image contains explicit reaction conditions but no R-group tables or sets of product variants (the R-group agents already interpret conditions internally).',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
        },
        {
        'type': 'function',
        'function': {
            'name': 'text_extraction_agent',
            'description': 'Extract the text from the image.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
        },
    ]

    with open('./prompt/prompt_final_simple_version.txt', 'r', encoding='utf-8') as prompt_file:
    #with open('./prompt/prompt_data.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    with open('./prompt/prompt_plan.txt', 'r', encoding='utf-8') as prompt_file:
        planner_user_message = prompt_file.read()

    planner_response = client.chat.completions.create(
        model='gpt-5-mini',
        messages=[
            {'role': 'system', 'content': "You are a chemical image understanding and extraction planning expert. After checking the image, your ONLY task is to SELECT and CALL the most appropriate agents from the list below to best fit the data extraction of the image."},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': planner_user_message},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
                ]
            }
        ]
    )
    
    planner_output = planner_response.choices[0].message.content.strip()
    print(f"[D] Planner output: {planner_output}")
    
    agent_list = _parse_planner_output(planner_output)
    print(f"[D] Parsed agents: {agent_list}")
    
    if use_plan_observer:
        observer_output = plan_observer_agent_azure(image_path, agent_list)
        reviewed = observer_output.get("list_of_agents", agent_list)
        reason = observer_output.get("reason", "")
        if isinstance(reviewed, list) and reviewed:
            new_agents = []
            for item in reviewed:
                if isinstance(item, str):
                    new_agents.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("tool_name") or ""
                    if name:
                        new_agents.append(name)
            if new_agents:
                agent_list = new_agents
                print(f"[D] Plan observer revised agents: {agent_list}")
                if reason:
                    print(f"[D] Plan observer reason: {reason}")
    

    ordered_agents, has_text_extraction = _resolve_ordered_agents(agent_list)

    AGENT_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group_azure,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group_azure,
        'get_full_reaction_template': get_full_reaction_template_azure,
        'get_multi_molecular_full': get_multi_molecular_full_azure,
        'get_reaction_con': get_reaction_con_azure,
        'text_extraction_agent': text_extraction_agent_azure
    }

    execution_logs = []
    results = []
    main_area_result = None
    observer_notes = []
    failed_agents = []

    def _observe_and_retry_azure(observed_name, observed_result, rerun):
        """Per-agent Action Observer check; at most one re-execution on redo.
        The observer's diagnosis is collected and forwarded to the final
        synthesis step."""
        check = action_observer_agent_azure(
            image_path, [{"name": observed_name, "result": observed_result}])
        if check.get("redo"):
            reason = check.get("reason", "")
            observer_notes.append(f"{observed_name}: {reason}" if reason else observed_name)
            print(f"[D] Action observer requested redo for {observed_name}: {reason}")
            observed_result = rerun()
        return observed_result

    for idx, agent_name in enumerate(ordered_agents):
        print(f"[D] Executing agent {idx + 1}/{len(ordered_agents)}: {agent_name}")
        try:
            agent_result = AGENT_MAP[agent_name](image_path=image_path)
            if use_action_observer:
                agent_result = _observe_and_retry_azure(
                    agent_name, agent_result,
                    lambda: AGENT_MAP[agent_name](image_path=image_path))
        except Exception as exc:
            failed_agents.append(f"{agent_name}: {type(exc).__name__}: {exc}")
            observer_notes.append(f"{agent_name} failed and was skipped ({type(exc).__name__})")
            print(f"[D] Agent {agent_name} failed, continuing without it: {exc!r}")
            continue
        if main_area_result is None:
            main_area_result = agent_result
        execution_logs.append({
            "id": f"agent_call_{idx}",
            "name": agent_name,
            "arguments": {"image_path": image_path},
            "result": agent_result,
        })
        results.append({
            'role': 'tool',
            'content': json.dumps({
                'image_path': image_path,
                agent_name: agent_result,
            }),
            'tool_call_id': f"agent_call_{idx}",
        })

    if not results:
        raise RuntimeError(
            "All planned agents failed, nothing left to synthesise: "
            + " | ".join(failed_agents)
        )

    observer_reason = "; ".join(observer_notes)

    text_extraction_result = None
    if has_text_extraction:
        print(f"[D] Executing text_extraction_agent with graphical_input")
        try:
            text_extraction_result = text_extraction_agent_azure(
                image_path=image_path,
                graphical_input=main_area_result,
            )
            if use_action_observer and text_extraction_result is not None:
                text_extraction_result = _observe_and_retry_azure(
                    "text_extraction_agent", text_extraction_result,
                    lambda: text_extraction_agent_azure(
                        image_path=image_path, graphical_input=main_area_result))
        except Exception as exc:

            failed_agents.append(f"text_extraction_agent: {type(exc).__name__}: {exc}")
            observer_notes.append(f"text_extraction_agent failed and was skipped ({type(exc).__name__})")
            print(f"[D] Agent text_extraction_agent failed, continuing without it: {exc!r}")
            text_extraction_result = None
        observer_reason = "; ".join(observer_notes)


    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": res['tool_call_id'],
                "type": "function",
                "function": {
                    "name": log["name"],
                    "arguments": json.dumps({"image_path": image_path}),
                },
            }
            for log, res in zip(execution_logs, results)
        ],
    }

    messages_list = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{base64_image}'
                    }
                }
            ]
        },
        assistant_message,
        *results,
    ]

    if text_extraction_result is not None:
        if isinstance(text_extraction_result, dict) and "annotated_text" in text_extraction_result:
            _text_extraction_for_msg = {"annotated_text": text_extraction_result["annotated_text"]}
        else:
            _text_extraction_for_msg = text_extraction_result
        messages_list.append({
        "role": "user",
        "content": (
            "Additionally, the text_extraction_agent has produced the following "
            "JSON for the prose portion of the same image."
            "```json\n"
            + json.dumps(_text_extraction_for_msg, ensure_ascii=False, indent=2)
            + "\n```"
        ),
    })
    
    if observer_reason:
        messages_list.append({
            'role': 'user',
            'content': f"Note: the previous execution had potential errors: {observer_reason}. Please review the results carefully.",
        })

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages_list,
        response_format={ 'type': 'json_object' },
        temperature=0,
    )

    gpt_output = json.loads(response.choices[0].message.content)
    gpt_output = fallback_validate_and_fix_smiles_in_dict(gpt_output)
    gpt_output = fallback_resolve_condition_smiles_in_data(gpt_output)
    gpt_output = fallback_resolve_reactant_product_smiles_in_data(gpt_output)    


    
    if text_extraction_result is not None:
        if isinstance(text_extraction_result, dict) and "annotated_text" in text_extraction_result:
            gpt_output["text_extraction"] = [{"annotated_text": text_extraction_result["annotated_text"]}]
        else:
            gpt_output["text_extraction"] = [text_extraction_result]
        
    print(gpt_output)
    return gpt_output