import sys
import torch
import json
from chemietoolkit import ChemIEToolkit
import cv2
from PIL import Image
import json
import sys
#sys.path.append('./RxnScribe-main/')
import torch
from rxnscribe import RxnScribe
import json
from molscribe.chemistry import _convert_graph_to_smiles

from openai import AzureOpenAI
import base64
import numpy as np
from chemietoolkit import utils
from PIL import Image




ckpt_path = "./pix2seq_reaction_full.ckpt"
model1 = RxnScribe(ckpt_path, device=torch.device('cpu'))
device = torch.device('cpu')
model = ChemIEToolkit(device=torch.device('cpu')) 

def get_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molscribe=True, ocr=True)

    # Ensure raw_prediction is treated as a list directly
    structured_output = {}
    for section_key in ['reactants', 'conditions', 'products']:
        if section_key in raw_prediction[0]:
            structured_output[section_key] = []
            for item in raw_prediction[0][section_key]:
                if section_key in ['reactants', 'products']:
                    # Extract smiles and bbox for molecules
                    structured_output[section_key].append({
                        "smiles": item.get("smiles", ""),
                        "bbox": item.get("bbox", []),
                        "symbols": item.get("symbols", [])  
                    })
                elif section_key == 'conditions':
                    # Extract smiles, text, and bbox for conditions
                    condition_data = {"bbox": item.get("bbox", [])}
                    if "smiles" in item:
                        condition_data["smiles"] = item.get("smiles", "")
                    if "text" in item:
                        condition_data["text"] = item.get("text", [])
                    structured_output[section_key].append(condition_data)
    print(structured_output)

    return structured_output



def get_full_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molscribe=True, ocr=True)
    return raw_prediction



def get_reaction_withatoms(image_path: str) -> dict:
    """

    Args:
        image_path (str): 图像文件路径。

    Returns:
        dict: 整理后的反应数据，包括反应物、产物和反应模板。
    """
    # 配置 API Key 和 Azure Endpoint
    api_key = "b038da96509b4009be931e035435e022"  # 替换为实际的 API Key
    azure_endpoint = "https://hkust.azure-api.net"  # 替换为实际的 Azure Endpoint
    
    model = ChemIEToolkit(device=torch.device('cpu'))
    client = AzureOpenAI(
        api_key=api_key,
        api_version='2024-06-01',
        azure_endpoint=azure_endpoint
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
            'name': 'get_reaction',
            'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
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
    with open('./prompt_getreaction.txt', 'r') as prompt_file:
        prompt = prompt_file.read()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    # 调用 GPT 接口
    response = client.chat.completions.create(
    model = 'gpt-4o',
    temperature = 0,
    response_format={ 'type': 'json_object' },
    messages = [
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
            ]},
    ],
    tools = tools)
    
# Step 1: 工具映射表
    TOOL_MAP = {
        'get_reaction': get_reaction,
    }

    # Step 2: 处理多个工具调用
    tool_calls = response.choices[0].message.tool_calls
    results = []

    # 遍历每个工具调用
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        tool_args = json.loads(tool_arguments)
        
        if tool_name in TOOL_MAP:
            # 调用工具并获取结果
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        
        # 保存每个工具调用结果
        results.append({
            'role': 'tool',
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}':(tool_result),
            }),
            'tool_call_id': tool_call_id,
        })


# Prepare the chat completion payload
    completion_payload = {
        'model': 'gpt-4o',
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
            response.choices[0].message,
            *results
            ],
    }

# Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
        temperature=0
    )


    
    # 获取 GPT 生成的结果
    gpt_output = json.loads(response.choices[0].message.content)
    print(f"gpt_output1:{gpt_output}")

    
    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''
        image_file = image_path
        raw_prediction = model1.predict_image_file(image_file, molscribe=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)



    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'products']:
            for item in input1.get(key, []):
                bbox = tuple(item['bbox'])  # 使用 bbox 作为唯一标识
                symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'products']:
            for item in input2.get(key, []):
                bbox = tuple(item['bbox'])  # 获取 bbox 作为匹配键

                # 如果 bbox 存在于 input1 的映射中，则更新 symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # 更新 atoms 的 atom_symbol
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # 如果 coords 和 edges 存在，调用转换函数生成新的 smiles 和 molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        
                        # 替换旧的 smiles 和 molfile
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [update_input_with_symbols(gpt_output, input2[0], _convert_graph_to_smiles)]

    return updated_data

 


def get_reaction_withatoms_correctR(image_path: str) -> dict:
    """

    Args:
        image_path (str): 图像文件路径。

    Returns:
        dict: 整理后的反应数据，包括反应物、产物和反应模板。
    """
    # 配置 API Key 和 Azure Endpoint
    api_key = "b038da96509b4009be931e035435e022"  # 替换为实际的 API Key
    azure_endpoint = "https://hkust.azure-api.net"  # 替换为实际的 Azure Endpoint

    model = ChemIEToolkit(device=torch.device('cpu'))
    client = AzureOpenAI(
        api_key=api_key,
        api_version='2024-06-01',
        azure_endpoint=azure_endpoint
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
            'name': 'get_reaction',
            'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
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
    with open('./prompt_getreaction_correctR.txt', 'r') as prompt_file:
        prompt = prompt_file.read()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    # 调用 GPT 接口
    response = client.chat.completions.create(
    model = 'gpt-4o',
    temperature = 0,
    response_format={ 'type': 'json_object' },
    messages = [
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
            ]},
    ],
    tools = tools)
    
# Step 1: 工具映射表
    TOOL_MAP = {
        'get_reaction': get_reaction,
    }

    # Step 2: 处理多个工具调用
    tool_calls = response.choices[0].message.tool_calls
    results = []

    # 遍历每个工具调用
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        tool_args = json.loads(tool_arguments)
        
        if tool_name in TOOL_MAP:
            # 调用工具并获取结果
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        
        # 保存每个工具调用结果
        results.append({
            'role': 'tool',
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}':(tool_result),
            }),
            'tool_call_id': tool_call_id,
        })


# Prepare the chat completion payload
    completion_payload = {
        'model': 'gpt-4o',
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
            response.choices[0].message,
            *results
            ],
    }

# Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
        temperature=0
    )


    
    # 获取 GPT 生成的结果
    gpt_output = json.loads(response.choices[0].message.content)
    print(f"gpt_output1:{gpt_output}")

    
    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''
        image_file = image_path
        raw_prediction = model1.predict_image_file(image_file, molscribe=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)



    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'products']:
            for item in input1.get(key, []):
                bbox = tuple(item['bbox'])  # 使用 bbox 作为唯一标识
                symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'products']:
            for item in input2.get(key, []):
                bbox = tuple(item['bbox'])  # 获取 bbox 作为匹配键

                # 如果 bbox 存在于 input1 的映射中，则更新 symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # 更新 atoms 的 atom_symbol
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # 如果 coords 和 edges 存在，调用转换函数生成新的 smiles 和 molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        
                        # 替换旧的 smiles 和 molfile
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [update_input_with_symbols(gpt_output, input2[0], _convert_graph_to_smiles)]
    print(f"updated_reaction_data:{updated_data}")

    return updated_data