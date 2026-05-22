import sys
import torch
import json
from chemietoolkit import ChemIEToolkit
import cv2
from PIL import Image
import json
import sys
import torch
from rxnim import RxnIM
import json
from molnextr.chemistry import _convert_graph_to_smiles

from openai import AzureOpenAI, OpenAI, InternalServerError, RateLimitError, APIError
import base64
import numpy as np
from chemietoolkit import utils
from PIL import Image
import os
from typing import Optional
import time


def retry_api_call(func, max_retries=3, base_delay=2, backoff_factor=2, *args, **kwargs):
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (InternalServerError, RateLimitError, APIError) as e:
            last_exception = e
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            error_message = str(e)
            
            # Check whether this is a 503 error or another retryable error
            if error_code == 503 or 'overloaded' in error_message.lower() or '503' in error_message:
                if attempt < max_retries - 1:
                    delay = base_delay * (backoff_factor ** attempt)
                    print(f"⚠️ API call failed (503/overloaded), attempt {attempt + 1}/{max_retries}. Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ API call failed, reached maximum retries ({max_retries})")
                    raise
            else:
                # Other error types, raise directly
                raise
        except Exception as e:
            # Other unknown errors, raise directly
            raise
    
    # If all retries failed
    if last_exception:
        raise last_exception
    raise RuntimeError("API call failed, unknown error")

ckpt_path = "./rxn.ckpt"
model1 = RxnIM(ckpt_path, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model = ChemIEToolkit(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")


def get_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image_file = image_path
    image = Image.open(image_file)

    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    #print(f'raw_prediction:{raw_prediction}')

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
                        condition_data["symbols"] = item.get("symbols", [])
                    if "text" in item:
                        condition_data["text"] = item.get("text", [])
                    structured_output[section_key].append(condition_data)
    #print(f'structured_output:{structured_output}')

    return structured_output



def get_full_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    for reaction in raw_prediction:
        for section in ("reactants", "products", "conditions"):
            for entry in reaction.get(section, []):
                # 1) Keep coords to three decimal places
                coords = entry.get("coords")
                if isinstance(coords, list):
                    entry["coords"] = [
                        [round(val, 3) for val in point]
                        for point in coords
                    ]
                # 2) Remove unnecessary fields
                for key in ("molfile", "atoms", "bonds"):
                    entry.pop(key, None)

    raw_prediction =json.dumps(raw_prediction)
    return raw_prediction



def get_reaction_withatoms(image_path: str) -> dict:
    """
    Input a chemical reaction image path, use GPT and OpenChemIE to extract reaction information, and return organized reaction data.

    Args:
        image_path (str): image file path.

    Returns:
        dict: organized reaction data, including reactants, products, and reaction templates.
    """
    # Initialize OpenChemIE model and Azure OpenAI client
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    # Load image and encode as Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT tool-calling configuration
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

    # Message content provided to GPT
    with open('./prompt/prompt_getreaction.txt', 'r', encoding='utf-8') as prompt_file:
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

    # Call GPT API
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
    
# Step 1: Tool mapping table
    TOOL_MAP = {
        'get_reaction': get_reaction,
    }

    # Step 2: Handle multiple tool calls
    tool_calls = response.choices[0].message.tool_calls
    results = []

    # Iterate through each tool call
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        tool_args = json.loads(tool_arguments)
        
        if tool_name in TOOL_MAP:
            # Call tool and get result
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        
        # Save each tool-call result
        results.append({
            'role': 'tool',
            'name': tool_name,  # Gemini API requires the name field
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


    
    # Get GPT-generated result
    gpt_output = json.loads(response.choices[0].message.content)
    #print(f"gpt_output1:{gpt_output}")

    
    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''
        image_file = image_path
        raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)



    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'products']:
            for item in input1.get(key, []):
                bbox = tuple(item['bbox'])  # Use bbox as a unique identifier
                symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'products']:
            for item in input2.get(key, []):
                bbox = tuple(item['bbox'])  # Get bbox as matching key

                # If bbox exists in input1 mapping, update symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # Update atom_symbol in atoms
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # If coords and edges exist, call conversion function to generate new smiles and molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        
                        # Replace old smiles and molfile
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [update_input_with_symbols(gpt_output, input2[0], _convert_graph_to_smiles)]

    return updated_data

 


def get_reaction_withatoms_correctR(image_path: str) -> dict:
    """
    Input a chemical reaction image path, use GPT and OpenChemIE to extract reaction information, and return organized reaction data.

    Args:
        image_path (str): image file path.

    Returns:
        dict: organized reaction data, including reactants, products, and reaction templates.
    """
    # Configure API Key and Azure Endpoint
    

    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    # Load image and encode as Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT tool-calling configuration
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

    # Message content provided to GPT
    with open('./prompt/prompt_Rxn_Tem.txt', 'r', encoding='utf-8') as prompt_file:
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

    # Call GPT API
    response = client.chat.completions.create(
    model = 'gpt-5-mini',
    #temperature = 0,
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
    
# Step 1: Tool mapping table
    TOOL_MAP = {
        'get_reaction': get_reaction,
    }

    # Step 2: Handle multiple tool calls
    tool_calls = response.choices[0].message.tool_calls
    results = []

    # Iterate through each tool call
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        tool_args = json.loads(tool_arguments)
        
        if tool_name in TOOL_MAP:
            # Call tool and get result
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        
        # Save each tool-call result
        results.append({
            'role': 'tool',
            'name': tool_name,  # Gemini API requires the name field
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}':(tool_result),
            }),
            'tool_call_id': tool_call_id,
        })


# Prepare the chat completion payload
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
            response.choices[0].message,
            *results
            ],
    }

# Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
        #temperature=0
    )


    
    # Get GPT-generated result
    gpt_output = json.loads(response.choices[0].message.content)
    print(f"gpt_output_rxn:{gpt_output}")

    
    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''

        image_file = image_path
        raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)



    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'conditions', 'products']:
            for item in input1.get(key, []):
                # Only handle items with symbols and bbox fields (conditions may only have text without symbols)
                if 'symbols' in item and 'bbox' in item:
                    bbox = tuple(item['bbox'])  # Use bbox as a unique identifier
                    symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'conditions', 'products']:
            for item in input2.get(key, []):
                if 'bbox' not in item:
                    continue
                bbox = tuple(item['bbox'])  # Get bbox as matching key

                # If bbox exists in input1 mapping, update symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # Update atom_symbol in atoms
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # If coords and edges exist, call conversion function to generate new smiles and molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        
                        # Replace old smiles and molfile
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [update_input_with_symbols(gpt_output, input2[0], _convert_graph_to_smiles)]
    print(f"rxn_agent_output:{updated_data}")

    return updated_data


def get_reaction_withatoms_correctR_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
 

    base_url = base_url or os.getenv("VLLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:8000/v1"))
    api_key = api_key or os.getenv("VLLM_API_KEY", os.getenv("OLLAMA_API_KEY", "EMPTY"))

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Load image and encode as Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT tool-calling configuration
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

    # Message content provided to GPT
    with open('./prompt/prompt_Rxn_Tem.txt', 'r', encoding='utf-8') as prompt_file:
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

    # Call GPT API (with retry mechanism)
    response = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=model_name,
        temperature=0,
        #response_format={'type': 'json_object'},  # vLLM does not support using response_format and tools simultaneously
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    # Step 1: Tool mapping table
    TOOL_MAP = {
        'get_reaction': get_reaction,
    }

    # Step 2: Handle multiple tool calls
    tool_calls = response.choices[0].message.tool_calls or []
    results = []

    # Iterate through each tool call
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        tool_args = json.loads(tool_arguments)
        
        if tool_name in TOOL_MAP:
            # Call tool and get result
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        
        # Save each tool-call result
        results.append({
            'role': 'tool',
            'name': tool_name,  # Gemini API requires the name field
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}':(tool_result),
            }),
            'tool_call_id': tool_call_id,
        })

    # Prepare the chat completion payload
    completion_payload = {
        'model': model_name,
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

    # Generate new response (with retry mechanism)
    response = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        #response_format={'type': 'json_object'},  # vLLM may not support this
        temperature=0
    )

    # Get GPT-generated result (supports extraction from text containing reasoning)
    from get_R_group_sub_agent import extract_json_from_text_with_reasoning
    
    raw_content = response.choices[0].message.content
    
    try:
        # First try direct parsing
        gpt_output = json.loads(raw_content)
        print(f"DEBUG [OS]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        # If direct parsing fails, use intelligent extraction function
        print(f"WARNING [OS]: Direct JSON parsing failed, trying to extract JSON from text...")
        gpt_output = extract_json_from_text_with_reasoning(raw_content)
        
        if gpt_output is not None:
            print(f"DEBUG [OS]: Successfully extracted JSON from text (with reasoning support)")
        else:
            print(f"ERROR [OS]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            raise json.JSONDecodeError(
                f"Could not parse JSON from model response. Content may not be valid JSON.",
                raw_content, 0
            )
    
    print(f"gpt_output_rxn:{gpt_output}")

    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''

        image_file = image_path
        raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)

    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'conditions', 'products']:
            for item in input1.get(key, []):
                # Only handle items with symbols and bbox fields (conditions may only have text without symbols)
                if 'symbols' in item and 'bbox' in item:
                    bbox = tuple(item['bbox'])  # Use bbox as a unique identifier
                    symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'conditions', 'products']:
            for item in input2.get(key, []):
                if 'bbox' not in item:
                    continue
                bbox = tuple(item['bbox'])  # Get bbox as matching key

                # If bbox exists in input1 mapping, update symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # Update atom_symbol in atoms
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # If coords and edges exist, call conversion function to generate new smiles and molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        
                        # Replace old smiles and molfile
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [update_input_with_symbols(gpt_output, input2[0], _convert_graph_to_smiles)]
    print(f"rxn_agent_output:{updated_data}")

    return updated_data

def _get_extra_body(model_name: str) -> dict:
    if "Qwen3.5" in model_name or "qwen3.5" in model_name:
        return {"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.05}
    return {}

def _tesseract_ocr_image(image_path: str) -> str:
    import pytesseract
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)
    return raw_text


def get_reaction_c(image_path: str) -> dict:
    raw_prediction = model1.predict_image_file(image_path, molnextr=True, ocr=True)
    conditions_per_reaction = []
    for reaction in raw_prediction:
        conds = reaction.get('conditions', [])
        cleaned = []
        for item in conds:
            cleaned.append({
                'text': item.get('text', ''),
                'category': item.get('category', ''),
                'bbox': item.get('bbox', []),
            })
        conditions_per_reaction.append(cleaned)
    return {'conditions': conditions_per_reaction}


def get_reaction_con(image_path: str) -> dict:
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )

    def encode_image(p: str):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'TesseractOCR',
                'description': 'Run Tesseract OCR on the reaction image and return the extracted raw text (including all condition texts).',
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
                'name': 'get_reaction_c',
                'description': 'RxnConInterpreter: extract and initially classify reaction condition texts (reagent/solvent/temperature/yield/etc.) from the reaction image.',
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

    with open('./prompt/prompt_con.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}},
            ],
        },
    ]

    response = client.chat.completions.create(
        model='gpt-5-mini',
        response_format={'type': 'json_object'},
        messages=messages,
        tools=tools,
    )

    TOOL_MAP = {
        'TesseractOCR': _tesseract_ocr_image,
        'get_reaction_c': get_reaction_c,
    }

    tool_calls = response.choices[0].message.tool_calls or []
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_call_id = tool_call.id
        if tool_name in TOOL_MAP:
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        results.append({
            'role': 'tool',
            'name': tool_name,
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}': tool_result,
            }),
            'tool_call_id': tool_call_id,
        })

    completion_payload = {
        'model': 'gpt-5-mini',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}},
                ],
            },
            response.choices[0].message,
            *results,
        ],
    }

    response = client.chat.completions.create(
        model=completion_payload['model'],
        messages=completion_payload['messages'],
        response_format={'type': 'json_object'},
    )

    gpt_output = json.loads(response.choices[0].message.content)
    print(f"gpt_output_con:{gpt_output}")
    return gpt_output


def get_reaction_con_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    base_url = base_url or os.getenv("VLLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:8000/v1"))
    api_key = api_key or os.getenv("VLLM_API_KEY", os.getenv("OLLAMA_API_KEY", "EMPTY"))

    client = OpenAI(base_url=base_url, api_key=api_key)

    def encode_image(p: str):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'TesseractOCR',
                'description': 'Run Tesseract OCR on the reaction image and return the extracted raw text (including all condition texts).',
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
                'name': 'get_reaction_c',
                'description': 'RxnConInterpreter: extract and initially classify reaction condition texts (reagent/solvent/temperature/yield/etc.) from the reaction image.',
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

    with open('./prompt/prompt_con.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}},
            ],
        },
    ]

    response = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=model_name,
        temperature=0,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        extra_body=_get_extra_body(model_name),
    )

    TOOL_MAP = {
        'TesseractOCR': _tesseract_ocr_image,
        'get_reaction_c': get_reaction_c,
    }

    tool_calls = response.choices[0].message.tool_calls or []
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_call_id = tool_call.id
        if tool_name in TOOL_MAP:
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        results.append({
            'role': 'tool',
            'name': tool_name,
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}': tool_result,
            }),
            'tool_call_id': tool_call_id,
        })

    completion_payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}},
                ],
            },
            response.choices[0].message,
            *results,
        ],
    }

    response = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=completion_payload['model'],
        messages=completion_payload['messages'],
        temperature=0,
        extra_body=_get_extra_body(model_name),
    )

    from get_R_group_sub_agent import extract_json_from_text_with_reasoning

    raw_content = response.choices[0].message.content
    try:
        gpt_output = json.loads(raw_content)
        print(f"DEBUG [con OS]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        print(f"WARNING [con OS]: Direct JSON parsing failed, trying to extract JSON from text...")
        gpt_output = extract_json_from_text_with_reasoning(raw_content)
        if gpt_output is not None:
            print(f"DEBUG [con OS]: Successfully extracted JSON from text (with reasoning support)")
        else:
            print(f"ERROR [con OS]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            raise json.JSONDecodeError(
                "Could not parse JSON from model response. Content may not be valid JSON.",
                raw_content, 0,
            )

    print(f"gpt_output_con:{gpt_output}")
    return gpt_output
