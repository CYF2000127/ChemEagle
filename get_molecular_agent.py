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
import sys
import torch
import json
model = ChemIEToolkit(device=torch.device('cpu')) 
from molscribe.chemistry import _convert_graph_to_smiles
import base64
import torch
import json
from PIL import Image
import numpy as np
from chemietoolkit import ChemIEToolkit, utils
from openai import AzureOpenAI





ckpt_path = "./pix2seq_reaction_full.ckpt"
model1 = RxnScribe(ckpt_path, device=torch.device('cpu'))
device = torch.device('cpu')

model = ChemIEToolkit(device=torch.device('cpu')) 

def get_multi_molecular(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    
    # 将图像作为输入传递给模型
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs']: #'atoms'
                bbox.pop(key, None)  # 安全地移除键
    print(json.dumps(coref_results))
    # 返回反应列表，使用 json.dumps 进行格式化
    
    return json.dumps(coref_results)

def get_multi_molecular_text_to_correct(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    
    # 将图像作为输入传递给模型
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "bbox", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs']: #'atoms'
                bbox.pop(key, None)  # 安全地移除键
    print(json.dumps(coref_results))
    # 返回反应列表，使用 json.dumps 进行格式化
    
    return json.dumps(coref_results)

def get_multi_molecular_text_to_correct_withatoms(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    
    # 将图像作为输入传递给模型
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["coords","edges","molfile", 'atoms', "bonds", 'category_id', 'score', 'corefs']: #'atoms'
                bbox.pop(key, None)  # 安全地移除键
    print(json.dumps(coref_results))
    # 返回反应列表，使用 json.dumps 进行格式化
    return json.dumps(coref_results)






def process_reaction_image_with_multiple_products_and_text(image_path: str) -> dict:
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
            'name': 'get_multi_molecular_text_to_correct_withatoms',
            'description': 'Extracts the SMILES string, the symbols set, and the text coref of all molecular images in a table-reaction image and ready to be correct.',
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
    with open('./prompt_getmolecular.txt', 'r') as prompt_file:
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
        'get_multi_molecular_text_to_correct_withatoms': get_multi_molecular_text_to_correct_withatoms,
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
    gpt_output = [json.loads(response.choices[0].message.content)]


    def get_multi_molecular(image_path: str) -> list:
        '''Returns a list of reactions extracted from the image.'''
        # 打开图像文件
        image = Image.open(image_path).convert('RGB')
        
        # 将图像作为输入传递给模型
        coref_results = model.extract_molecule_corefs_from_figures([image])
        return coref_results

    
    coref_results = get_multi_molecular(image_path)


    def update_symbols_in_atoms(input1, input2):
        """
        用 input1 中更新后的 'symbols' 替换 input2 中对应 bboxes 的 'symbols'，并同步更新 'atoms' 的 'atom_symbol'。
        假设 input1 和 input2 的结构一致。
        """
        for item1, item2 in zip(input1, input2):
            bboxes1 = item1.get('bboxes', [])
            bboxes2 = item2.get('bboxes', [])
            
            if len(bboxes1) != len(bboxes2):
                print("Warning: Mismatched number of bboxes!")
                continue

            for bbox1, bbox2 in zip(bboxes1, bboxes2):
                # 更新 symbols
                if 'symbols' in bbox1:
                    bbox2['symbols'] = bbox1['symbols']  # 更新 symbols
                
                # 更新 atoms 的 atom_symbol
                if 'symbols' in bbox1 and 'atoms' in bbox2:
                    symbols = bbox1['symbols']
                    atoms = bbox2.get('atoms', [])
                    
                    # 确保 symbols 和 atoms 的长度一致
                    if len(symbols) != len(atoms):
                        print(f"Warning: Mismatched symbols and atoms in bbox {bbox1.get('bbox')}!")
                        continue

                    for atom, symbol in zip(atoms, symbols):
                        atom['atom_symbol'] = symbol  # 更新 atom_symbol

        return input2


    input2_updated = update_symbols_in_atoms(gpt_output, coref_results)





    def update_smiles_and_molfile(input_data, conversion_function):
        """
        使用更新后的 'symbols'、'coords' 和 'edges' 调用 `conversion_function` 生成新的 'smiles' 和 'molfile'，
        并替换到原数据结构中。
        
        参数:
        - input_data: 包含 bboxes 的嵌套数据结构
        - conversion_function: 函数，接受 'coords', 'symbols', 'edges' 并返回 (new_smiles, new_molfile, _)
        
        返回:
        - 更新后的数据结构
        """
        for item in input_data:
            for bbox in item.get('bboxes', []):
                # 检查必需的键是否存在
                if all(key in bbox for key in ['coords', 'symbols', 'edges']):
                    coords = bbox['coords']
                    symbols = bbox['symbols']
                    edges = bbox['edges']
                    
                    # 调用转换函数生成新的 'smiles' 和 'molfile'
                    new_smiles, new_molfile, _ = conversion_function(coords, symbols, edges)
                    print(f"    Generated 'smiles': {new_smiles}")
            
                    # 替换旧的 'smiles' 和 'molfile'
                    bbox['smiles'] = new_smiles
                    bbox['molfile'] = new_molfile

        return input_data

    updated_data = update_smiles_and_molfile(input2_updated, _convert_graph_to_smiles)

    return updated_data

    
    






def process_reaction_image_with_multiple_products_and_text_correctR(image_path: str) -> dict:
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
            'name': 'get_multi_molecular_text_to_correct_withatoms',
            'description': 'Extracts the SMILES string, the symbols set, and the text coref of all molecular images in a table-reaction image and ready to be correct.',
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
    with open('./prompt_getmolecular_correctR.txt', 'r') as prompt_file:
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
        'get_multi_molecular_text_to_correct_withatoms': get_multi_molecular_text_to_correct_withatoms,
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
    gpt_output = [json.loads(response.choices[0].message.content)]


    def get_multi_molecular(image_path: str) -> list:
        '''Returns a list of reactions extracted from the image.'''
        # 打开图像文件
        image = Image.open(image_path).convert('RGB')
        
        # 将图像作为输入传递给模型
        coref_results = model.extract_molecule_corefs_from_figures([image])
        return coref_results

    
    coref_results = get_multi_molecular(image_path)


    def update_symbols_in_atoms(input1, input2):
        """
        用 input1 中更新后的 'symbols' 替换 input2 中对应 bboxes 的 'symbols'，并同步更新 'atoms' 的 'atom_symbol'。
        假设 input1 和 input2 的结构一致。
        """
        for item1, item2 in zip(input1, input2):
            bboxes1 = item1.get('bboxes', [])
            bboxes2 = item2.get('bboxes', [])
            
            if len(bboxes1) != len(bboxes2):
                print("Warning: Mismatched number of bboxes!")
                continue

            for bbox1, bbox2 in zip(bboxes1, bboxes2):
                # 更新 symbols
                if 'symbols' in bbox1:
                    bbox2['symbols'] = bbox1['symbols']  # 更新 symbols
                
                # 更新 atoms 的 atom_symbol
                if 'symbols' in bbox1 and 'atoms' in bbox2:
                    symbols = bbox1['symbols']
                    atoms = bbox2.get('atoms', [])
                    
                    # 确保 symbols 和 atoms 的长度一致
                    if len(symbols) != len(atoms):
                        print(f"Warning: Mismatched symbols and atoms in bbox {bbox1.get('bbox')}!")
                        continue

                    for atom, symbol in zip(atoms, symbols):
                        atom['atom_symbol'] = symbol  # 更新 atom_symbol

        return input2


    input2_updated = update_symbols_in_atoms(gpt_output, coref_results)





    def update_smiles_and_molfile(input_data, conversion_function):
        """
        使用更新后的 'symbols'、'coords' 和 'edges' 调用 `conversion_function` 生成新的 'smiles' 和 'molfile'，
        并替换到原数据结构中。
        
        参数:
        - input_data: 包含 bboxes 的嵌套数据结构
        - conversion_function: 函数，接受 'coords', 'symbols', 'edges' 并返回 (new_smiles, new_molfile, _)
        
        返回:
        - 更新后的数据结构
        """
        for item in input_data:
            for bbox in item.get('bboxes', []):
                # 检查必需的键是否存在
                if all(key in bbox for key in ['coords', 'symbols', 'edges']):
                    coords = bbox['coords']
                    symbols = bbox['symbols']
                    edges = bbox['edges']
                    
                    # 调用转换函数生成新的 'smiles' 和 'molfile'
                    new_smiles, new_molfile, _ = conversion_function(coords, symbols, edges)
                    print(f"    Generated 'smiles': {new_smiles}")
            
                    # 替换旧的 'smiles' 和 'molfile'
                    bbox['smiles'] = new_smiles
                    bbox['molfile'] = new_molfile

        return input_data

    updated_data = update_smiles_and_molfile(input2_updated, _convert_graph_to_smiles)
    print(f"updated_mol_data:{updated_data}")

    return updated_data
