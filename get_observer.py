import base64
import json
import os
import time
from typing import Any, List, Optional

from openai import AzureOpenAI, OpenAI
from openai import InternalServerError, RateLimitError, APIError


API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = API_KEY or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = AZURE_ENDPOINT or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = API_VERSION or "2024-06-01"
        
        if not api_key or not azure_endpoint:
            return None
        
        _client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
    return _client

PLAN_PROMPT_TEMPLATE = """System Message: 
You are a plan observer. Given the graphic and the current list of agent calls (plan), decide whether the plan is sufficient.

User Message: 
Given the graphic and the current list of agent calls (plan), please recheck the component of the graphic and decide whether the corresponding agent is called and the plan is sufficient.
Valid agents:
1. Reaction template parsing agent
Parses the reaction scheme to identify reactants, products, and label mappings, and outputs a structured reaction template.

2. Molecular recognition agent
Detects other molecules in the graphic except in the reaction template, recognizes their structures, and returns normalized representations (e.g., SMILES, labels, positions).

3. Structure-based R-group substitution agent
Uses structure panels / variant images to extract R-group values and generate enumerated products from a core scaffold based on structural information.

4. Text-based R-group substitution agent
Reads R-group tables and enumerates products or substituents on top of a given core scaffold using text information.

5. Condition interpretation agent
Extracts and normalizes reaction conditions (catalysts, reagents, solvent, temperature, time, atmosphere, etc.) from the graphic.

6. Text extraction agent
Performs chmical NER and text-based reaction extraction on the text description.

If the plan is acceptable, return the original plan as-is.
If adjustments are required, provide the improved list of agents and briefly explain the changes.

Always respond in valid JSON with the structure:
{
  "list_of_agents": [...],   // final list of agent calls
  "redo": true/false,
  "reason": "If changed is true, give an explanation; otherwise leave blank."
}

Current plan (JSON):
{plan_json}
"""

ACTION_PROMPT_TEMPLATE = """System Message: 
You are an action observer. Your task is too observe the graphic and the current agent output, decide whether the agent must be rerun.

User Message: 
By observing the image and the current agent output, decide whether the agent must be rerun.
The main focus is on whether the SMILES is reasonable and effective. Is the condition or text classification correct?
If the outcome is acceptable, return redo=false.
If issues are found or corrections are needed, return redo=true with a short explanation.

Always respond in valid JSON with the structure:
{
  "redo": true/false,
  "reason": "Provide the reasons when redo is true; otherwise leave blank."
}

Current agent_result (JSON):
{result_json}
"""


def _encode_image(image_path: str) -> str | None:
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def plan_observer_agent(image_path: str, tool_calls: List[Any]) -> List[Any]:
    base64_image = _encode_image(image_path)
    plan_json = json.dumps(tool_calls or [], ensure_ascii=False, indent=2)
    prompt = PLAN_PROMPT_TEMPLATE.format(plan_json=plan_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        client = _get_client()
        if client is None:
            # 没有 API 配置，返回原始计划
            return tool_calls
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        # Support both "plan" and "list_of_agents" keys for compatibility
        return parsed.get("list_of_agents", parsed.get("plan", tool_calls))
    except Exception:
        return tool_calls


def action_observer_agent(image_path: str, tool_result: Any) -> bool:
    base64_image = _encode_image(image_path)
    result_json = json.dumps(tool_result, ensure_ascii=False, indent=2)
    prompt = ACTION_PROMPT_TEMPLATE.format(result_json=result_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        client = _get_client()
        if client is None:
            # 没有 API 配置，返回 False（不重做）
            return False
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return bool(parsed.get("redo", False))
    except Exception:
        return False


def retry_api_call(func, max_retries=3, base_delay=2, backoff_factor=2, *args, **kwargs):
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (InternalServerError, RateLimitError, APIError) as e:
            last_exception = e
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            error_message = str(e)
            
            # 检查是否是 503 错误或其他可重试的错误
            if error_code == 503 or 'overloaded' in error_message.lower() or '503' in error_message:
                if attempt < max_retries - 1:
                    delay = base_delay * (backoff_factor ** attempt)
                    print(f"⚠️ API 调用失败 (503/过载)，第 {attempt + 1}/{max_retries} 次尝试。{delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ API 调用失败，已达到最大重试次数 ({max_retries})")
                    raise
            else:
                # 其他类型的错误，直接抛出
                raise
        except Exception as e:
            # 其他未知错误，直接抛出
            raise
    
    # 如果所有重试都失败了
    if last_exception:
        raise last_exception
    raise RuntimeError("API 调用失败，未知错误")


def plan_observer_agent_OS(
    image_path: str,
    tool_calls: List[Any],
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Any]:

    base_url = base_url or os.getenv("VLLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:8000/v1"))
    api_key = api_key or os.getenv("VLLM_API_KEY", os.getenv("OLLAMA_API_KEY", "EMPTY"))

    client_os = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    base64_image = _encode_image(image_path)
    plan_json = json.dumps(tool_calls or [], ensure_ascii=False, indent=2)
    prompt = PLAN_PROMPT_TEMPLATE.format(plan_json=plan_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        # Note: vLLM may not support response_format
        response = retry_api_call(
            client_os.chat.completions.create,
            max_retries=5,
            base_delay=3,
            backoff_factor=2,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            # response_format={"type": "json_object"},  # vLLM 可能不支持
        )
        content = response.choices[0].message.content
        
        # Try to parse JSON directly
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from text
            try:
                from get_R_group_sub_agent import extract_json_from_text_with_reasoning
                parsed = extract_json_from_text_with_reasoning(content)
                if parsed is None:
                    raise ValueError("Failed to extract JSON from response")
            except (ImportError, ValueError):
                print(f"⚠️ 警告: plan_observer_agent_OS 无法解析 JSON，返回原始计划")
                return tool_calls
        
        # Support both "plan" and "list_of_agents" keys for compatibility
        return parsed.get("list_of_agents", parsed.get("plan", tool_calls))
    except Exception as e:
        print(f"⚠️ 警告: plan_observer_agent_OS 出错: {e}，返回原始计划")
        return tool_calls


def action_observer_agent_OS(
    image_path: str,
    tool_result: Any,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:

    base_url = base_url or os.getenv("VLLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:8000/v1"))
    api_key = api_key or os.getenv("VLLM_API_KEY", os.getenv("OLLAMA_API_KEY", "EMPTY"))

    client_os = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    base64_image = _encode_image(image_path)
    result_json = json.dumps(tool_result, ensure_ascii=False, indent=2)
    prompt = ACTION_PROMPT_TEMPLATE.format(result_json=result_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        # Note: vLLM may not support response_format
        response = retry_api_call(
            client_os.chat.completions.create,
            max_retries=5,
            base_delay=3,
            backoff_factor=2,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            # response_format={"type": "json_object"},  # vLLM 可能不支持
        )
        content = response.choices[0].message.content
        
        # Try to parse JSON directly
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from text
            try:
                from get_R_group_sub_agent import extract_json_from_text_with_reasoning
                parsed = extract_json_from_text_with_reasoning(content)
                if parsed is None:
                    raise ValueError("Failed to extract JSON from response")
            except (ImportError, ValueError):
                print(f"⚠️: action_observer_agent_OS 无法解析 JSON，返回 False（不重做）")
                return False
        
        return bool(parsed.get("redo", False))
    except Exception as e:
        print(f"⚠️: action_observer_agent_OS: {e}，False")
        return False
