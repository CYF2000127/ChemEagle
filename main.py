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
from get_molecular_agent import process_reaction_image_with_multiple_products_and_text_correctR, process_reaction_image_with_multiple_products_and_text_correctmultiR
from get_reaction_agent import get_reaction_withatoms_correctR
from get_R_group_sub_agent import process_reaction_image_with_table_R_group, process_reaction_image_with_product_variant_R_group,get_full_reaction_template_OS,get_full_reaction_template, get_multi_molecular_full,get_multi_molecular_full_OS, process_reaction_image_with_table_R_group_OS,process_reaction_image_with_product_variant_R_group_OS,get_full_reaction_OS,get_reaction_OS
from get_observer import action_observer_agent, plan_observer_agent,action_observer_agent_OS, plan_observer_agent_OS
from get_text_agent import text_extraction_agent, text_extraction_agent_OS
from chemietoolkit.helper import _clean_agent_name, _parse_planner_output, _select_main_area, _has_text_extraction

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
    Given a chemical reaction image path, extract reaction information
    using GPT models and tools, and return structured reaction data.
    Supports plan observer and action observer.

    Args:
        image_path (str): Path to the image file.
        use_plan_observer (bool): Whether to use plan observer to review the tool call plan.
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

    with open('./prompt/prompt_final_simple_version.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    with open('./prompt/prompt_plan.txt', 'r', encoding='utf-8') as prompt_file:
        planner_user_message = prompt_file.read()

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
    
    planner_output = planner_response.choices[0].message.content.strip()
    print(f"[D] Planner output: {planner_output}")
    
    agent_list = _parse_planner_output(planner_output)
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
    
    agent_names_lower = [agent.lower() for agent in agent_list]
    selected_area = _select_main_area(agent_names_lower)
    
    print(f"[D] Selected area: {selected_area}")
    
    AREA_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group,
        'get_full_reaction_template': get_full_reaction_template,
        'get_multi_molecular_full': get_multi_molecular_full,
        'text_extraction_agent': text_extraction_agent
    }
    
    has_text_extraction = _has_text_extraction(agent_names_lower)

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

    text_extraction_result = None
    if has_text_extraction:
        print(f"[D] Executing text_extraction_agent with graphical_input")
        text_extraction_result = text_extraction_agent(
            image_path=image_path,
            graphical_input=main_area_result,
        )

    tool_call_id = results[0]['tool_call_id']
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": selected_area,
                    "arguments": json.dumps({"image_path": image_path}),
                },
            }
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
    if observer_reason:
        messages_list.append({
            'role': 'user',
            'content': f"Note: the previous execution had potential errors: {observer_reason}. Please review the results carefully.",
        })

    response = client.chat.completions.create(
        model='gpt-5-mini',
        messages=messages_list,
        response_format={ 'type': 'json_object' },
    )

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

    with open('./prompt/prompt_final_simple_version.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    with open('./prompt/prompt_plan.txt', 'r', encoding='utf-8') as prompt_file:
        planner_user_message = prompt_file.read()

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
    
    agent_list = _parse_planner_output(planner_output)
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
    
    agent_names_lower = [agent.lower() for agent in agent_list]
    selected_area = _select_main_area(agent_names_lower)
    
    print(f"[OS_D] Selected area: {selected_area}")
    
    AREA_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group_OS,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group_OS,
        'get_full_reaction_template': get_full_reaction_template_OS,
        'get_multi_molecular_full': get_multi_molecular_full_OS,
        'text_extraction_agent': text_extraction_agent_OS
    }
    
    has_text_extraction = _has_text_extraction(agent_names_lower)

    OS_TOOLS_ACCEPT_BASE_D = (
        "process_reaction_image_with_product_variant_R_group",
        "process_reaction_image_with_table_R_group",
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

    text_extraction_result = None
    if has_text_extraction:
        print(f"[OS_D] Executing text_extraction_agent with graphical_input")
        text_extraction_result = text_extraction_agent_OS(
            image_path=image_path,
            graphical_input=main_area_result,
            base_url=base_url,
            api_key=api_key,
        )

    tool_call_id = results[0]['tool_call_id']
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": selected_area,
                    "arguments": json.dumps({"image_path": image_path}),
                },
            }
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
    if observer_reason:
        messages_list.append({
            'role': 'user',
            'content': f"Note: the previous execution had potential errors: {observer_reason}. Please review the results carefully.",
        })

    response = client.chat.completions.create(
        model=model_name,
        messages=messages_list,
        temperature=0,
    )
    print(response)
    
    raw_content = response.choices[0].message.content
    
    from get_R_group_sub_agent import extract_json_from_text_with_reasoning
    
    try:
        gpt_output = json.loads(raw_content)
        print("DEBUG [OS_D]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        print("WARNING [OS_D]: Direct JSON parsing failed, trying to extract JSON from text...")
        gpt_output = extract_json_from_text_with_reasoning(raw_content)
        
        if gpt_output is not None:
            print("DEBUG [OS_D]: Successfully extracted JSON from text (with reasoning support)")
        else:
            print(f"ERROR [OS_D]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            print("WARNING [OS_D]: Returning tool results as fallback")
            tool_results_dict = {}
            for log in execution_logs:
                t_name = log.get("name")
                t_result = log.get("result")
                if t_name and t_name != "text_extraction_agent" and t_result is not None:
                    tool_results_dict[t_name] = t_result
            if len(tool_results_dict) == 1:
                single_result = list(tool_results_dict.values())[0]
                if isinstance(single_result, dict):
                    if text_extraction_result is not None:
                        single_result["text_extraction"] = text_extraction_result
                    return single_result
            else:
                if text_extraction_result is not None:
                    tool_results_dict["text_extraction"] = text_extraction_result
                return tool_results_dict

    if text_extraction_result is not None:
        gpt_output["text_extraction"] = text_extraction_result
    print(gpt_output)
    return gpt_output
