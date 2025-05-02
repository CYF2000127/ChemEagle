import sys
import torch
import json
from chemietoolkit import ChemIEToolkit,utils
import cv2
from openai import AzureOpenAI
import numpy as np
from PIL import Image
import json
from get_molecular_agent import process_reaction_image_with_multiple_products_and_text_correctR
from get_reaction_agent import get_reaction_withatoms_correctR
import sys
from rxnscribe import RxnScribe
import json
import base64
model = ChemIEToolkit(device=torch.device('cpu')) 
ckpt_path = "./pix2seq_reaction_full.ckpt"
model1 = RxnScribe(ckpt_path, device=torch.device('cpu'))
device = torch.device('cpu')


with open('api_key.txt', 'r') as api_key_file:
    API_KEY = api_key_file.read()

def parse_coref_data_with_fallback(data):
    bboxes = data["bboxes"]
    corefs = data["corefs"]
    paired_indices = set()

    # 先处理有 coref 配对的
    results = []
    for idx1, idx2 in corefs:
        smiles_entry = bboxes[idx1] if "smiles" in bboxes[idx1] else bboxes[idx2]
        text_entry = bboxes[idx2] if "text" in bboxes[idx2] else bboxes[idx1]

        smiles = smiles_entry.get("smiles", "")
        texts = text_entry.get("text", [])

        results.append({
            "smiles": smiles,
            "texts": texts
        })

        # 记录下哪些 SMILES 被配对过了
        paired_indices.add(idx1)
        paired_indices.add(idx2)

    # 处理未配对的 SMILES（补充进来）
    for idx, entry in enumerate(bboxes):
        if "smiles" in entry and idx not in paired_indices:
            results.append({
                "smiles": entry["smiles"],
                "texts": ["There is no label or failed to detect, please recheck the image again"]
            })

    return results
    

def get_multi_molecular_text_to_correct(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    
    # 将图像作为输入传递给模型
    #coref_results = process_reaction_image_with_multiple_products_and_text_correctR(image_path)
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "bbox", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # 安全地移除键

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)

    
    print(f"coref_results:{json.dumps(parsed)}")
    return json.dumps(parsed)








def get_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image, 
    including only reactants, conditions, and products with their smiles, bbox, or text.
    '''
    image_file = image_path
    #raw_prediction = model1.predict_image_file(image_file, molscribe=True, ocr=True)
    raw_prediction = get_reaction_withatoms_correctR(image_path)


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
                        "bbox": item.get("bbox", [])
                    })
                elif section_key == 'conditions':
                    # Extract text and bbox for conditions
                    structured_output[section_key].append({
                        "text": item.get("text", []),
                        "bbox": item.get("bbox", []),
                        "smiles": item.get("smiles", []),
                    })
    
    return structured_output



def process_reaction_image(image_path: str) -> dict:
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
                'name': 'get_multi_molecular_text_to_correct',
                'description': 'Extracts the SMILES string and text coref from molecular images.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'image_path': {
                            'type': 'string',
                            'description': 'Path to the reaction image.'
                        }
                    },
                    'required': ['image_path'],
                    'additionalProperties': False
                }
            }
        },
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
    with open('./prompt.txt', 'r') as prompt_file:
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
        'get_multi_molecular_text_to_correct': get_multi_molecular_text_to_correct,
        'get_reaction': get_reaction
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
    print(gpt_output)
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)


    # reaction_results = model.extract_reactions_from_figures([image_np])
    coref_results = model.extract_molecule_corefs_from_figures([image_np])

    reaction_results = get_reaction_withatoms_correctR(image_path)[0]
    reaction = {
    "reactants": reaction_results.get('reactants', []),
    "conditions": reaction_results.get('conditions', []),
    "products": reaction_results.get('products', [])
    }
    reaction_results = [{"reactions": [reaction]}]
    print(reaction_results)
    #coref_results = process_reaction_image_with_multiple_products_and_text_correctR(image_path)
    

    # 定义更新工具输出的函数
    def extract_smiles_details(smiles_data, raw_details):
        smiles_details = {}
        for smiles in smiles_data:
            for detail in raw_details:
                for bbox in detail.get('bboxes', []):
                    if bbox.get('smiles') == smiles:
                        smiles_details[smiles] = {
                            'category': bbox.get('category'),
                            'bbox': bbox.get('bbox'),
                            'category_id': bbox.get('category_id'),
                            'score': bbox.get('score'),
                            'molfile': bbox.get('molfile'),
                            'atoms': bbox.get('atoms'),
                            'bonds': bbox.get('bonds'),
                        }
                        break
        return smiles_details

# 获取结果
    smiles_details = extract_smiles_details(gpt_output, coref_results)

    reactants_array = []
    products = []

    for reactant in reaction_results[0]['reactions'][0]['reactants']:
        if 'smiles' in reactant:
            print(f"SMILES:{reactant['smiles']}")
            #print(reactant)
            reactants_array.append(reactant['smiles'])

    for product in reaction_results[0]['reactions'][0]['products']:
        #print(product['smiles'])
        #print(product)
        products.append(product['smiles'])
    # 输出结果
    #import pprint
    #pprint.pprint(smiles_details)

        # 整理反应数据
    backed_out = utils.backout_without_coref(reaction_results, coref_results, gpt_output, smiles_details, model.molscribe)
    backed_out.sort(key=lambda x: x[2])
    extracted_rxns = {}
    for reactants, products_, label in backed_out:
        extracted_rxns[label] = {'reactants': reactants, 'products': products_}

    toadd = {
        "reaction_template": {
            "reactants": reactants_array,
            "products": products
        },
        "reactions": extracted_rxns,
        "original_molecule_list": gpt_output
    }

# 按标签排序
    sorted_keys = sorted(toadd["reactions"].keys())
    toadd["reactions"] = {i: toadd["reactions"][i] for i in sorted_keys}
    print(toadd)
    return toadd




def ChemEagle(image_path: str) -> dict:
    """
    输入化学反应图像路径，通过 GPT 模型和 TOOLS 提取反应信息并返回整理后的反应数据。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        dict: 整理后的反应数据，包括反应物、产物和反应模板。
    """
    # 配置 API Key 和 Azure Endpoint
    api_key = API_KEY # 替换为实际的 API Key
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
            'name': 'process_reaction_image',
            'description': 'get the reaction data of the reaction diagram and get SMILES strings of every detailed reaction in reaction diagram and the table, and the original molecular list.',
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
    with open('./prompt_final_simple_version.txt', 'r') as prompt_file:
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
        'process_reaction_image': process_reaction_image
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
    print(gpt_output)
    return gpt_output