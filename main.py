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
from typing import Optional
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
    use_plan_observer: bool = True,
    use_action_observer: bool = True,
) -> dict:
    """
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

    with open('./prompt/prompt_plan_new.txt', 'r', encoding='utf-8') as prompt_file:
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

    planner_output = re.sub(r'[{}]', '', planner_output).strip()
    agent_list = [agent.strip() for agent in planner_output.split(',') if agent.strip()]
    print(f"[D] Parsed agents: {agent_list}")
    
    
    selected_tool = None
    agent_names_lower = [agent.lower() for agent in agent_list]
    
    if "structure-based r-group substitution agent" in agent_names_lower:
        selected_tool = "process_reaction_image_with_product_variant_R_group"
    elif "text-based r-group substitution agent" in agent_names_lower:
        selected_tool = "process_reaction_image_with_table_R_group"
    elif "reaction template parsing agent" in agent_names_lower:
        selected_tool = "get_full_reaction_template"
    elif "molecular recognition agent" in agent_names_lower:
        selected_tool = "get_multi_molecular_full"
    else:
        print(f"warning: no agents")
        selected_tool = "get_full_reaction_template"
    
    print(f"[D] Selected tool: {selected_tool}")
    
    TOOL_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group,
        'get_full_reaction_template': get_full_reaction_template,
        'get_multi_molecular_full': get_multi_molecular_full,
        'text_extraction_agent': text_extraction_agent
    }
    

    has_text_extraction = "text extraction agent" in agent_names_lower or "text_extraction_agent" in agent_names_lower
    
    serialized_calls = [{
        "id": "tool_call_0",
        "name": selected_tool,
        "arguments": {"image_path": image_path}
    }]
    
    if has_text_extraction:
        serialized_calls.append({
            "id": "tool_call_1",
            "name": "text_extraction_agent",
            "arguments": {"image_path": image_path}
        })
        print(f"[D] Added text_extraction_agent as second tool")
    
    # Plan Observer
    if use_plan_observer:
        reviewed_plan = plan_observer_agent(image_path, serialized_calls)
        if not isinstance(reviewed_plan, list) or not reviewed_plan:
            plan_to_execute = serialized_calls
        else:
            plan_to_execute = []
            for idx, item in enumerate(reviewed_plan):
                name = item.get("name") or item.get("tool_name")
                if not name:
                    continue
                args = item.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_id = item.get("id") or f"observer_call_{idx}"
                plan_to_execute.append({
                    "id": call_id,
                    "name": name,
                    "arguments": args,
                })
            if not plan_to_execute:
                plan_to_execute = serialized_calls
    else:
        plan_to_execute = serialized_calls

    print(f"[D] plan_to_execute:{plan_to_execute}")
    execution_logs = []
    results = []

    for idx, plan_item in enumerate(plan_to_execute):
        tool_name = plan_item.get("name") or plan_item.get("tool_name")
        if not tool_name:
            print(f"warning: plan_item {idx} no name ，skip: {plan_item}")
            continue
        
        tool_call_id = plan_item.get("id") or f"observer_call_{idx}"
        tool_args = _normalize_tool_args(plan_item.get("arguments", {}), image_path)

        if tool_name in TOOL_MAP:
            tool_func = TOOL_MAP[tool_name]
            tool_result = tool_func(**tool_args)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")

        execution_logs.append({
            "id": tool_call_id,
            "name": tool_name,
            "arguments": tool_args,
            "result": tool_result,
        })

        results.append({
            'role': 'tool',
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}':(tool_result),
            }),
            'tool_call_id': tool_call_id,
        })

    # Action Observer
    if use_action_observer and action_observer_agent(image_path, execution_logs):
        return {
            "redo": True,
            "plan": plan_to_execute,
            "execution_logs": execution_logs,
        }

    executed_tools = [selected_tool]
    if has_text_extraction:
        executed_tools.append("text_extraction_agent")
    assistant_message = {
        "role": "assistant",
        "content": f"Selected agents: {', '.join(agent_list)}\nExecuted tools: {', '.join(executed_tools)}"
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

    gpt_output = json.loads(response.choices[0].message.content)
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
    """
    Open source version of ChemEagle
    """
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

    with open('./prompt/prompt_plan_new.txt', 'r', encoding='utf-8') as prompt_file:
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
    
    planner_output = re.sub(r'[{}]', '', planner_output).strip()
    agent_list = [agent.strip() for agent in planner_output.split(',') if agent.strip()]
    print(f"[OS_D] Parsed agents: {agent_list}")
    
    selected_tool = None
    agent_names_lower = [agent.lower() for agent in agent_list]
    
    if "structure-based r-group substitution agent" in agent_names_lower:
        selected_tool = "process_reaction_image_with_product_variant_R_group"
    elif "text-based r-group substitution agent" in agent_names_lower:
        selected_tool = "process_reaction_image_with_table_R_group"
    elif "reaction template parsing agent" in agent_names_lower:
        selected_tool = "get_full_reaction_template"
    elif "molecular recognition agent" in agent_names_lower:
        selected_tool = "get_multi_molecular_full"
    else:
        print(f"warning: no agents")
        selected_tool = "get_full_reaction_template"
    
    print(f"[OS_D] Selected tool: {selected_tool}")
    
    TOOL_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group_OS,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group_OS,
        'get_full_reaction_template': get_full_reaction_template_OS,
        'get_multi_molecular_full': get_multi_molecular_full_OS,
        'text_extraction_agent': text_extraction_agent_OS
    }
    
    has_text_extraction = "text extraction agent" in agent_names_lower or "text_extraction_agent" in agent_names_lower
    
    serialized_calls = [{
        "id": "tool_call_0",
        "name": selected_tool,
        "arguments": {"image_path": image_path}
    }]
    
    if has_text_extraction:
        serialized_calls.append({
            "id": "tool_call_1",
            "name": "text_extraction_agent",
            "arguments": {"image_path": image_path}
        })
        print(f"[OS_D] Added text_extraction_agent as second tool")
    
    # Plan Observer
    if use_plan_observer:
        reviewed_plan = plan_observer_agent_OS(image_path, serialized_calls)
        if not isinstance(reviewed_plan, list) or not reviewed_plan:
            plan_to_execute = serialized_calls
        else:
            plan_to_execute = []
            for idx, item in enumerate(reviewed_plan):
                name = item.get("name") or item.get("tool_name")
                if not name:
                    continue
                args = item.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_id = item.get("id") or f"observer_call_{idx}"
                plan_to_execute.append({
                    "id": call_id,
                    "name": name,
                    "arguments": args,
                })
            if not plan_to_execute:
                plan_to_execute = serialized_calls
    else:
        plan_to_execute = serialized_calls

    print(f"[OS_D] plan_to_execute:{plan_to_execute}")
    execution_logs = []
    results = []

    for idx, plan_item in enumerate(plan_to_execute):
        tool_name = plan_item.get("name") or plan_item.get("tool_name")
        if not tool_name:
            print(f"warning: plan_item {idx} no name ，skip: {plan_item}")
            continue
        
        tool_call_id = plan_item.get("id") or f"observer_call_{idx}"
        tool_args = _normalize_tool_args(plan_item.get("arguments", {}), image_path)

        if tool_name in TOOL_MAP:
            tool_func = TOOL_MAP[tool_name]
            tool_result = tool_func(**tool_args)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")

        execution_logs.append({
            "id": tool_call_id,
            "name": tool_name,
            "arguments": tool_args,
            "result": tool_result,
        })

        if not tool_name or not tool_name.strip():
            print(f"warning: tool_name is empty，skip")
            continue
            
        results.append({
            'role': 'tool',
            'name': tool_name.strip(),
            'content': json.dumps({
                'image_path': image_path,
                tool_name: tool_result,
            }),
            'tool_call_id': tool_call_id,
        })
    
    print(f'[OS_D] results: {results}')
    
    # Action Observer
    if use_action_observer and action_observer_agent_OS(image_path, execution_logs):
        return {
            "redo": True,
            "plan": plan_to_execute,
            "execution_logs": execution_logs,
        }

    executed_tools = [selected_tool]
    if has_text_extraction:
        executed_tools.append("text_extraction_agent")
    assistant_message = {
        "role": "assistant",
        "content": f"Selected agents: {', '.join(agent_list)}\nExecuted tools: {', '.join(executed_tools)}"
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
            print("WARNING [OS_D]: Returning raw content as fallback")
            return {"content": raw_content, "parsed": False}
    
    print(gpt_output)
    return gpt_output



if __name__ == "__main__":
    model = ChemEagle()
