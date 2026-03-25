import sys
import torch
import json
from chemietoolkit import ChemIEToolkit,utils
import cv2
from openai import AzureOpenAI, OpenAI
import numpy as np
from PIL import Image
import json
import os
import sys
from rxnim import RxnIM
import json
import base64
import re
from typing import Optional, Dict, Any, List
from get_molecular_agent import process_reaction_image_with_multiple_products_and_text_correctR, process_reaction_image_with_multiple_products_and_text_correctmultiR
from get_reaction_agent import get_reaction_withatoms_correctR
from get_R_group_sub_agent import process_reaction_image_with_table_R_group, process_reaction_image_with_product_variant_R_group,get_full_reaction_template_OS,get_full_reaction_template, get_multi_molecular_full,get_multi_molecular_full_OS, process_reaction_image_with_table_R_group_OS,process_reaction_image_with_product_variant_R_group_OS,get_full_reaction_OS,get_reaction_OS
from get_observer import action_observer_agent, plan_observer_agent,action_observer_agent_OS, plan_observer_agent_OS
from get_text_agent import text_extraction_agent, text_extraction_agent_OS


model = ChemIEToolkit(device=torch.device('cpu')) 
ckpt_path = "./rxn.ckpt"
model1 = RxnIM(ckpt_path, device=torch.device('cpu'))
device = torch.device('cpu')

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

def _normalize_tool_args(raw_args: Optional[dict], image_path: str) -> dict:
    if not isinstance(raw_args, dict):
        return {"image_path": image_path}
    normalized = dict(raw_args)
    placeholder_values = {"[img]", "<img>", "[image]", "<image>", "<<<IMAGE>>>", "IMAGE_PATH", "image.png","image_path"}
    if normalized.get("image_path") in placeholder_values or normalized.get("image_path") is None:
        normalized["image_path"] = image_path
    return normalized


def ChemEagle(
    image_path: str,
    *,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
) -> dict:
    """
    输入化学反应图像路径，通过 GPT 模型和 TOOLS 提取反应信息并返回整理后的反应数据。
    这是 ChemEagle 的增强版本，支持 plan observer 和 action observer。

    Args:
        image_path (str): 图像文件路径。
        use_plan_observer (bool): 是否使用 plan observer 来审查和修改工具调用计划。
        use_action_observer (bool): 是否使用 action observer 来检查执行结果，如果失败则重新执行。

    Returns:
        dict: 整理后的反应数据，包括反应物、产物和反应模板。
    """
    # 初始化 Azure OpenAI 客户端
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    # 加载图像并编码为 Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT 工具调用配置
    tools = [
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
            'description': 'After you carefully check the image, if this is a reaction image that contains only a text-based table and does not involve any R-group replacement, or this is a reaction image does not contain any tables or sets of product variants, then just call this simplified tool.',
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
            'description': 'After you carefully check the image, if this is a single molecule image or a multiple molecules image, then need to call this molecular recognition tool.',
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

    # 提供给 GPT 的消息内容
    with open('./prompt/prompt_final_simple_version.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    with open('./prompt/prompt_plan.txt', 'r', encoding='utf-8') as prompt_file:
        planner_user_message = prompt_file.read()

    # Step 1: 调用 planner 获取 agent 列表
    planner_response = client.chat.completions.create(
        model='gpt-5-mini',
        messages=[
            {'role': 'system', 'content': "You are a chemical image understanding and extraction planning expert.After checking the image, your ONLY task is to SELECT and CALL the most appropriate agents from the list below to best fit the data extraction of the image."},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': planner_user_message},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
                ]
            }
        ]
    )
    
    # 解析 planner 返回的 agent 列表
    planner_output = planner_response.choices[0].message.content.strip()
    print(f"[D] Planner output: {planner_output}")
    
    # 提取 agent 名称（移除可能的括号、花括号等）
    # 移除 { } 和多余的空白
    planner_output = re.sub(r'[{}]', '', planner_output).strip()
    # 分割为 agent 列表
    agent_list = [agent.strip() for agent in planner_output.split(',') if agent.strip()]
    print(f"[D] Parsed agents: {agent_list}")
    
    if use_plan_observer:
        observer_output = plan_observer_agent(image_path, agent_list)
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
    
    # Step 3: agent 名称 → 工具函数名映射
    selected_area = None
    agent_names_lower = [agent.lower() for agent in agent_list]
    
    if "structure-based r-group substitution agent" in agent_names_lower:
        selected_area = "process_reaction_image_with_product_variant_R_group"
    elif "text-based r-group substitution agent" in agent_names_lower:
        selected_area = "process_reaction_image_with_table_R_group"
    elif "reaction template parsing agent" in agent_names_lower:
        selected_area = "get_full_reaction_template"
    elif "molecular recognition agent" in agent_names_lower:
        selected_area = "get_multi_molecular_full"
    else:
        print(f"warning: no agents")
        selected_area = "get_full_reaction_template"
    
    print(f"[D] Selected area: {selected_area}")
    
    AREA_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group,
        'get_full_reaction_template': get_full_reaction_template,
        'get_multi_molecular_full': get_multi_molecular_full,
        'text_extraction_agent': text_extraction_agent
    }
    
    # Step 4: 执行主 area
    has_text_extraction = "text extraction agent" in agent_names_lower or "text_extraction_agent" in agent_names_lower

    print(f"[D] Executing main area: {selected_area}")
    main_area_result = AREA_MAP[selected_area](image_path=image_path)
    execution_logs = [{
        "id": "tool_call_0",
        "name": selected_area,
        "arguments": {"image_path": image_path},
        "result": main_area_result,
    }]
    results = [{
        'role': 'tool',
        'content': json.dumps({
            'image_path': image_path,
            selected_area: main_area_result,
        }),
        'tool_call_id': "tool_call_0",
    }]

    # Action Observer: 仅检查主 area 执行结果
    observer_reason = ""
    if use_action_observer:
        observer_result = action_observer_agent(image_path, execution_logs)
        if observer_result.get("redo"):
            observer_reason = observer_result.get("reason", "")
            print(f"[D] Action observer requested redo: {observer_reason}")
            main_area_result = AREA_MAP[selected_area](image_path=image_path)
            execution_logs[0] = {
                "id": "retry_call_0",
                "name": selected_area,
                "arguments": {"image_path": image_path},
                "result": main_area_result,
            }
            results[0] = {
                'role': 'tool',
                'content': json.dumps({
                    'image_path': image_path,
                    selected_area: main_area_result,
                }),
                'tool_call_id': "retry_call_0",
            }

    # 执行 text_extraction_agent（传入最终的主 area 结果，结果不传给第二次 LLM）
    text_extraction_result = None
    if has_text_extraction:
        print(f"[D] Executing text_extraction_agent with graphical_input")
        text_extraction_result = text_extraction_agent(
            image_path=image_path,
            graphical_input=main_area_result,
        )

    # 构建 assistant 消息（仅包含主 area，text_extraction 不传给 LLM）
    msg = f"Executed areas: {selected_area}"
    if observer_reason:
        msg += f"\nPotential error from previous execution: {observer_reason}"
    assistant_message = {
        "role": "assistant",
        "content": msg
    }
    
    completion_payload = {
        'model': 'gpt-5-mini',
        'messages': [
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
            *results
            ],
    }

    # Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
    )

    # 获取 GPT 生成的结果
    gpt_output = json.loads(response.choices[0].message.content)
    if text_extraction_result is not None:
        gpt_output["text_extraction"] = text_extraction_result
    print(gpt_output)
    return gpt_output



def ChemEagle_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
) -> dict:

    base_url = base_url or os.getenv("VLLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:8000/v1"))
    api_key = api_key or os.getenv("VLLM_API_KEY", os.getenv("OLLAMA_API_KEY", "EMPTY"))

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    def encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    base64_image = encode_image(image_path)

    # 提供给 GPT 的消息内容
    with open('./prompt/prompt_final_simple_version.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    with open('./prompt/prompt_plan.txt', 'r', encoding='utf-8') as prompt_file:
        planner_user_message = prompt_file.read()

    # Step 1: 调用 planner 获取 agent 列表
    planner_response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {'role': 'system', 'content': "You are a chemical image understanding and extraction planning expert.After checking the image, your ONLY task is to SELECT and CALL the most appropriate agents from the list below to best fit the data extraction of the image."},
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
    print(f"[OS_D] Planner output: {planner_output}")
    
    planner_output = re.sub(r'[{}]', '', planner_output).strip()
    # 分割为 agent 列表
    agent_list = [agent.strip() for agent in planner_output.split(',') if agent.strip()]
    print(f"[OS_D] Parsed agents: {agent_list}")
    
    if use_plan_observer:
        observer_output = plan_observer_agent_OS(image_path, agent_list, model_name=model_name, base_url=base_url, api_key=api_key)
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
                print(f"[OS_D] Plan observer revised agents: {agent_list}")
                if reason:
                    print(f"[OS_D] Plan observer reason: {reason}")
    
    selected_area = None
    agent_names_lower = [agent.lower() for agent in agent_list]
    
    if "structure-based r-group substitution agent" in agent_names_lower:
        selected_area = "process_reaction_image_with_product_variant_R_group"
    elif "text-based r-group substitution agent" in agent_names_lower:
        selected_area = "process_reaction_image_with_table_R_group"
    elif "reaction template parsing agent" in agent_names_lower:
        selected_area = "get_full_reaction_template"
    elif "molecular recognition agent" in agent_names_lower:
        selected_area = "get_multi_molecular_full"
    else:
        print(f"warning: no agents")
        selected_area = "get_full_reaction_template"
    
    print(f"[OS_D] Selected area: {selected_area}")
    
    AREA_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group_OS,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group_OS,
        'get_full_reaction_template': get_full_reaction_template_OS,
        'get_multi_molecular_full': get_multi_molecular_full_OS,
        'text_extraction_agent': text_extraction_agent_OS
    }
    
    has_text_extraction = "text extraction agent" in agent_names_lower or "text_extraction_agent" in agent_names_lower

    OS_TOOLS_ACCEPT_BASE_D = (
        "process_reaction_image_with_product_variant_R_group",
        "process_reaction_image_with_table_R_group",
        "get_full_reaction_template",
        "get_multi_molecular_full",
    )

    print(f"[OS_D] Executing main area: {selected_area}")
    main_area_args = {"image_path": image_path}
    if selected_area in OS_TOOLS_ACCEPT_BASE_D:
        main_area_args["base_url"] = base_url
        main_area_args["api_key"] = api_key
    main_area_result = AREA_MAP[selected_area](**main_area_args)

    execution_logs = [{
        "id": "tool_call_0",
        "name": selected_area,
        "arguments": {"image_path": image_path},
        "result": main_area_result,
    }]
    results = [{
        'role': 'tool',
        'name': selected_area,
        'content': json.dumps({
            'image_path': image_path,
            selected_area: main_area_result,
        }),
        'tool_call_id': "tool_call_0",
    }]

    print(f'[OS_D] results: {results}')

    observer_reason = ""
    if use_action_observer:
        observer_result = action_observer_agent_OS(image_path, execution_logs, model_name=model_name, base_url=base_url, api_key=api_key)
        if observer_result.get("redo"):
            observer_reason = observer_result.get("reason", "")
            print(f"[OS_D] Action observer requested redo: {observer_reason}")
            retry_args = {"image_path": image_path}
            if selected_area in OS_TOOLS_ACCEPT_BASE_D:
                retry_args["base_url"] = base_url
                retry_args["api_key"] = api_key
            main_area_result = AREA_MAP[selected_area](**retry_args)
            execution_logs[0] = {
                "id": "retry_call_0",
                "name": selected_area,
                "arguments": {"image_path": image_path},
                "result": main_area_result,
            }
            results[0] = {
                'role': 'tool',
                'name': selected_area,
                'content': json.dumps({
                    'image_path': image_path,
                    selected_area: main_area_result,
                }),
                'tool_call_id': "retry_call_0",
            }

    # 执行 text_extraction_agent（传入最终的主 area 结果，结果不传给第二次 LLM）
    text_extraction_result = None
    if has_text_extraction:
        print(f"[OS_D] Executing text_extraction_agent with graphical_input")
        text_extraction_result = text_extraction_agent_OS(
            image_path=image_path,
            graphical_input=main_area_result,
            base_url=base_url,
            api_key=api_key,
        )

    # 构建 assistant 消息（仅包含主 area，text_extraction 不传给 LLM）
    msg = f"Executed areas: {selected_area}"
    if observer_reason:
        msg += f"\nPotential error from previous execution: {observer_reason}"
    assistant_message = {
        "role": "assistant",
        "content": msg
    }
    
    completion_payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}' }}
                ],
            },
            assistant_message,
            *results
            ],
    }

    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        #response_format={ 'type': 'json_object' },
        temperature=0,
    )
    print(response)
    
    # 获取原始响应内容
    raw_content = response.choices[0].message.content
    
    # 尝试解析 JSON（支持从包含思考过程的文本中提取）
    from get_R_group_sub_agent import extract_json_from_text_with_reasoning
    
    try:
        # 首先尝试直接解析
        gpt_output = json.loads(raw_content)
        print("DEBUG [OS_D]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        # 如果直接解析失败，使用智能提取函数（支持思考模型输出）
        print("WARNING [OS_D]: Direct JSON parsing failed, trying to extract JSON from text...")
        gpt_output = extract_json_from_text_with_reasoning(raw_content)
        
        if gpt_output is not None:
            print("DEBUG [OS_D]: Successfully extracted JSON from text (with reasoning support)")
        else:
            print(f"ERROR [OS_D]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            # 如果无法解析为 JSON，直接返回工具结果
            print("WARNING [OS_D]: Returning tool results as fallback")
            # 从 execution_logs 中提取主 area 工具结果（排除 text_extraction）
            tool_results_dict = {}
            for log in execution_logs:
                t_name = log.get("name")
                t_result = log.get("result")
                if t_name and t_name != "text_extraction_agent" and t_result is not None:
                    tool_results_dict[t_name] = t_result
            if len(tool_results_dict) == 1:
                single_result = list(tool_results_dict.values())[0]
                if isinstance(single_result, dict):
                    single_result = single_result
                    if text_extraction_result is not None:
                        single_result["text_extraction"] = text_extraction_result
                return single_result
            else:
                for t_name, t_result in tool_results_dict.items():
                    if isinstance(t_result, dict):
                        tool_results_dict[t_name] = t_result
                if text_extraction_result is not None:
                    tool_results_dict["text_extraction"] = text_extraction_result
                return tool_results_dict

    if text_extraction_result is not None:
        gpt_output["text_extraction"] = text_extraction_result
    print(gpt_output)
    return gpt_output


if __name__ == "__main__":
    model = ChemEagle()
