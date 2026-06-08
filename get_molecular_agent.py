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
import sys
import torch
import json 
from molnextr.chemistry import _convert_graph_to_smiles
import base64
import torch
import json
from PIL import Image
import numpy as np
from chemietoolkit import ChemIEToolkit, utils
from openai import AzureOpenAI, OpenAI, InternalServerError, RateLimitError, APIError
import os
import copy
from typing import Optional
import time
from chemietoolkit.helper import _patch_to_mol


def _ga_bbox_iou(a, b) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    if not (a and b and len(a) == 4 and len(b) == 4):
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ga_best_iou_match_idx(target_bbox, orig_bboxes, iou_thresh=0.5):
    """Index of the original bbox with the highest IoU vs ``target_bbox``.

    Template reuse is allowed (the molecular flow expands one core into several
    variants that all echo the same template bbox), so this does NOT exclude
    already-claimed indices. Returns ``None`` below ``iou_thresh``.
    """
    best_idx, best_iou = None, 0.0
    for i, bb in enumerate(orig_bboxes):
        iou = _ga_bbox_iou(target_bbox, bb.get("bbox"))
        if iou > best_iou:
            best_iou, best_idx = iou, i
    if best_idx is not None and best_iou >= iou_thresh:
        return best_idx
    return None



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

def get_multi_molecular(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # Open image file
    image = Image.open(image_path).convert('RGB')
    
    # Pass image as input to the model
    coref_results = model.extract_molecule_corefs_from_figures([image])
    #print(f"coref_results:{coref_results}")
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs']: #'atoms'
                bbox.pop(key, None)  # Safely remove key
    #print(json.dumps(coref_results))
    # Return reaction list, formatted with json.dumps
    
    return json.dumps(coref_results)

def get_multi_molecular_text_to_correct(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # Open image file
    image = Image.open(image_path).convert('RGB')
    
    # Pass image as input to the model
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "bbox", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs']: #'atoms'
                bbox.pop(key, None)  # Safely remove key
    #print(json.dumps(coref_results))
    # Return reaction list, formatted with json.dumps
    
    return json.dumps(coref_results)

def get_multi_molecular_text_to_correct_withatoms(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # Open image file
    image = Image.open(image_path).convert('RGB')
    
    # Pass image as input to the model
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["coords","edges","molfile", 'atoms', "bonds", 'category_id', 'score', 'corefs']: #'atoms'
                bbox.pop(key, None)  # Safely remove key
    #print(json.dumps(coref_results))
    # Return reaction list, formatted with json.dumps
    return json.dumps(coref_results)






def process_reaction_image_with_multiple_products_and_text(image_path: str) -> dict:
    """


    Args:
        image_path (str): image file path.

    Returns:
        dict: organized reaction data, including reactants, products, and reaction templates.
    """

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

    # Message content provided to GPT
    with open('./prompt/prompt_getmolecular.txt', 'r', encoding='utf-8') as prompt_file:
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
        'get_multi_molecular_text_to_correct_withatoms': get_multi_molecular_text_to_correct_withatoms,
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
    gpt_output = [json.loads(response.choices[0].message.content)]


    def get_multi_molecular(image_path: str) -> list:
        '''Returns a list of reactions extracted from the image.'''
        # Open image file
        image = Image.open(image_path).convert('RGB')
        
        # Pass image as input to the model
        coref_results = model.extract_molecule_corefs_from_figures([image])
        return coref_results

    
    coref_results = get_multi_molecular(image_path)


    def update_symbols_in_atoms(input1, input2):
        """
        Replace corresponding bbox symbols in input2 with updated symbols from input1, and synchronously update atom_symbol in atoms.
        Assume input1 and input2 have consistent structure.
        """
        for item1, item2 in zip(input1, input2):
            bboxes1 = item1.get('bboxes', [])
            bboxes2 = item2.get('bboxes', [])
            
            if len(bboxes1) != len(bboxes2):
                print("Warning: Mismatched number of bboxes!")
                continue

            for bbox1, bbox2 in zip(bboxes1, bboxes2):
                # Update symbols
                if 'symbols' in bbox1:
                    bbox2['symbols'] = bbox1['symbols']  # Update symbols
                
                # Update atom_symbol in atoms
                if 'symbols' in bbox1 and 'atoms' in bbox2:
                    symbols = bbox1['symbols']
                    atoms = bbox2.get('atoms', [])
                    
                    # Ensure symbols and atoms have consistent lengths
                    if len(symbols) != len(atoms):
                        print(f"Warning: Mismatched symbols and atoms in bbox {bbox1.get('bbox')}!")
                        continue

                    for atom, symbol in zip(atoms, symbols):
                        atom['atom_symbol'] = symbol  # Update atom_symbol

        return input2


    input2_updated = update_symbols_in_atoms(gpt_output, coref_results)





    def update_smiles_and_molfile(input_data, conversion_function):
        """
        Use updated symbols, coords, and edges to call `conversion_function` to generate new smiles and molfile,
        and replace them in the original data structure.
        
        Parameters:
        - input_data: nested data structure containing bboxes
        - conversion_function: function accepting coords, symbols, edges and returning (new_smiles, new_molfile, _)
        
        Returns:
        - updated data structure
        """
        for item in input_data:
            for bbox in item.get('bboxes', []):
                # Check whether required keys exist
                if all(key in bbox for key in ['coords', 'symbols', 'edges']):
                    coords = bbox['coords']
                    symbols = bbox['symbols']
                    edges = bbox['edges']
                    
                    # Call conversion function to generate new smiles and molfile
                    new_smiles, new_molfile, _ = conversion_function(coords, symbols, edges)
                    #print(f"    Generated 'smiles': {new_smiles}")
            
                    # Replace old 'smiles' and 'molfile'
                    bbox['smiles'] = new_smiles
                    bbox['molfile'] = new_molfile

        return input_data

    updated_data = update_smiles_and_molfile(input2_updated, _convert_graph_to_smiles)

    return updated_data

    
    






def process_reaction_image_with_multiple_products_and_text_correctR(image_path: str) -> dict:
    """


    Args:
        image_path (str): image file path.

    Returns:
        dict: organized reaction data, including reactants, products, and reaction templates.
    """
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

    # Message content provided to GPT
    with open('./prompt/prompt_getmolecular_correctR.txt', 'r', encoding='utf-8') as prompt_file:
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
        'get_multi_molecular_text_to_correct_withatoms': get_multi_molecular_text_to_correct_withatoms,
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
    gpt_output = [json.loads(response.choices[0].message.content)]
    print(f"gpt_output_mol:{gpt_output}")


    def get_multi_molecular(image_path: str) -> list:
        '''Returns a list of reactions extracted from the image.'''
        # Open image file
        image = Image.open(image_path).convert('RGB')
        
        # Pass image as input to the model
        coref_results = model.extract_molecule_corefs_from_figures([image])
        return coref_results

    
    coref_results = get_multi_molecular(image_path)


    def update_symbols_in_atoms(input1, input2):
        """
        Replace corresponding bbox symbols in input2 with updated symbols from input1, and synchronously update atom_symbol in atoms.
        Assume input1 and input2 have consistent structure.
        """
        for item1, item2 in zip(input1, input2):
            bboxes1 = item1.get('bboxes', [])
            bboxes2 = item2.get('bboxes', [])
            
            if len(bboxes1) != len(bboxes2):
                print("Warning: Mismatched number of bboxes!")
                continue

            for bbox1, bbox2 in zip(bboxes1, bboxes2):
                # Update symbols
                if 'symbols' in bbox1:
                    bbox2['symbols'] = bbox1['symbols']  # Update symbols
                
                # Update atom_symbol in atoms
                if 'symbols' in bbox1 and 'atoms' in bbox2:
                    symbols = bbox1['symbols']
                    atoms = bbox2.get('atoms', [])
                    
                    # Ensure symbols and atoms have consistent lengths
                    if len(symbols) != len(atoms):
                        print(f"Warning: Mismatched symbols and atoms in bbox {bbox1.get('bbox')}!")
                        continue

                    for atom, symbol in zip(atoms, symbols):
                        atom['atom_symbol'] = symbol  # Update atom_symbol

        return input2


    input2_updated = update_symbols_in_atoms(gpt_output, coref_results)





    def update_smiles_and_molfile(input_data, conversion_function):
        """
        Use updated symbols, coords, and edges to call `conversion_function` to generate new smiles and molfile,
        and replace them in the original data structure.
        
        Parameters:
        - input_data: nested data structure containing bboxes
        - conversion_function: function accepting coords, symbols, edges and returning (new_smiles, new_molfile, _)
        
        Returns:
        - updated data structure
        """
        for item in input_data:
            for bbox in item.get('bboxes', []):
                # Check whether required keys exist
                if all(key in bbox for key in ['coords', 'symbols', 'edges']):
                    coords = bbox['coords']
                    symbols = bbox['symbols']
                    edges = bbox['edges']
                    
                    # Call conversion function to generate new smiles and molfile
                    new_smiles, new_molfile, _ = conversion_function(coords, symbols, edges)
                    #print(f"    Generated 'smiles': {new_smiles}")
            
                    # Replace old 'smiles' and 'molfile'
                    bbox['smiles'] = new_smiles
                    bbox['molfile'] = new_molfile

        return input_data

    updated_data = update_smiles_and_molfile(input2_updated, _convert_graph_to_smiles)
    print(f"mol_agent_output:{updated_data}")

    return updated_data



def process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path: str) -> dict:
    """


    Args:
        image_path (str): image file path.

    Returns:
        dict: organized reaction data, including reactants, products, and reaction templates.
    """
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

    # Message content provided to GPT
    with open('./prompt/prompt_Mol_Reco.txt', 'r', encoding='utf-8') as prompt_file:
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
        'get_multi_molecular_text_to_correct_withatoms': get_multi_molecular_text_to_correct_withatoms,
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
    gpt_output = [json.loads(response.choices[0].message.content)]
    print(f"gpt_output_mol:{gpt_output}")


    def get_multi_molecular(image_path: str) -> list:
        '''Returns a list of reactions extracted from the image.'''
        # Open image file
        image = Image.open(image_path).convert('RGB')
        
        # Pass image as input to the model
        coref_results = model.extract_molecule_corefs_from_figures([image])
        return coref_results

    
    coref_results = get_multi_molecular(image_path)


    def update_symbols_and_corefs(gpt_outputs, coref_results):
        results = []
        for item1, item2 in zip(gpt_outputs, coref_results):
            orig_bboxes = item2.get('bboxes', [])
            orig_corefs = item2.get('corefs', [])
            # 1. Construct new bboxes (prefer exact bbox template, fall back to best-IoU on drift)
            coord2idx = {tuple(bb['bbox']): i for i, bb in enumerate(orig_bboxes)}
            new_bboxes = []
            # Track which new bbox indices each original template expanded into,
            # so corefs can be rebuilt by index.
            orig2new = {}
            for bb1 in item1.get('bboxes', []):
                coord = tuple(bb1['bbox'])
                if coord in coord2idx:
                    tmpl_idx = coord2idx[coord]
                else:
                    tmpl_idx = _ga_best_iou_match_idx(bb1.get('bbox'), orig_bboxes)
                    if tmpl_idx is None:
                        print(f"WARNING [mol-agent]: bbox {coord} not matched to any original template, skipping it.")
                        continue
                bb_template = orig_bboxes[tmpl_idx]
                bb_new = copy.deepcopy(bb_template)
                if 'symbols' in bb1:
                    bb_new['symbols'] = bb1['symbols']
                    if 'atoms' in bb_new:
                        for atom, sym in zip(bb_new['atoms'], bb1['symbols']):
                            atom['atom_symbol'] = sym
                if 'text' in bb1:
                    bb_new['text'] = bb1['text']
                if 'sub_text' in bb1:
                    bb_new['sub_text'] = bb1['sub_text']
                bb_new['bbox'] = bb1['bbox']
                orig2new.setdefault(tmpl_idx, []).append(len(new_bboxes))
                new_bboxes.append(bb_new)

            # 2. Build corefs (rebuild via original-index -> new-index map; skip gracefully on dropped boxes)
            new_corefs = []
            for group in orig_corefs:
                # Assume group = [mol_idx, idt_idx] or [mol_idx1, mol_idx2, ..., idt_idx]
                label_idx = group[-1]
                label_new_list = orig2new.get(label_idx, [])
                if not label_new_list:
                    continue
                new_label_idx = label_new_list[-1]  # label has only one
                # All expanded new indices of mols
                for mol_idx in group[:-1]:
                    for new_mol_idx in orig2new.get(mol_idx, []):
                        new_corefs.append([new_mol_idx, new_label_idx])
            # 3. Assemble structure
            new_item = copy.deepcopy(item2)
            new_item['bboxes'] = new_bboxes
            new_item['corefs'] = new_corefs
            results.append(new_item)
        return results


    input2_updated = update_symbols_and_corefs(gpt_output, coref_results)

    def update_smiles_and_molfile(input_data, conversion_function):
        """
        Use updated symbols, coords, and edges to call `conversion_function` to generate new smiles and molfile,
        and replace them in the original data structure.
        
        Parameters:
        - input_data: nested data structure containing bboxes
        - conversion_function: function accepting coords, symbols, edges and returning (new_smiles, new_molfile, _)
        
        Returns:
        - updated data structure
        """
        for item in input_data:
            for bbox in item.get('bboxes', []):
                # Check whether required keys exist
                if all(key in bbox for key in ['coords', 'symbols', 'edges']):
                    coords = bbox['coords']
                    symbols = bbox['symbols']
                    edges = bbox['edges']
                    
                    # Call conversion function to generate new smiles and molfile
                    new_smiles, new_molfile, _ = conversion_function(coords, symbols, edges)
                    #print(f"    Generated 'smiles': {new_smiles}")
            
                    # Replace old 'smiles' and 'molfile'
                    bbox['smiles'] = new_smiles
                    bbox['molfile'] = new_molfile

        return input_data

    updated_data = update_smiles_and_molfile(input2_updated, _convert_graph_to_smiles)
    updated_data = _patch_to_mol(updated_data)
    print(f"mol_agent_output:{updated_data}")

    return updated_data


def process_reaction_image_with_multiple_products_and_text_correctmultiR_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Aligned with process_reaction_image_with_multiple_products_and_text_correctmultiR workflow, but uses a local/self-hosted model compatible with OpenAI Chat Completions protocol (such as vLLM or Ollama).

    Args:
        image_path: image file path.
        model_name: local model name (default `qwen3-vl:32b`).
        base_url: OpenAI-compatible API endpoint; if None, use `http://localhost:8000/v1` (vLLM default port).
        api_key: API key, can be any non-empty string (vLLM default can be `"EMPTY"`).

    Returns:
        dict: organized reaction data, including reactants, products, and reaction templates.
    """
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

    # Message content provided to GPT
    with open('./prompt/prompt_Mol_Reco.txt', 'r', encoding='utf-8') as prompt_file:
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
        'get_multi_molecular_text_to_correct_withatoms': get_multi_molecular_text_to_correct_withatoms,
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
        gpt_output = [json.loads(raw_content)]
        print(f"DEBUG [OS]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        # If direct parsing fails, use intelligent extraction function
        print(f"WARNING [OS]: Direct JSON parsing failed, trying to extract JSON from text...")
        parsed_json = extract_json_from_text_with_reasoning(raw_content)
        
        if parsed_json is not None:
            gpt_output = [parsed_json]
            print(f"DEBUG [OS]: Successfully extracted JSON from text (with reasoning support)")
        else:
            print(f"ERROR [OS]: Failed to parse JSON from model response")
            print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
            raise json.JSONDecodeError(
                f"Could not parse JSON from model response. Content may not be valid JSON.",
                raw_content, 0
            )
    
    print(f"gpt_output_mol:{gpt_output}")

    def get_multi_molecular(image_path: str) -> list:
        '''Returns a list of reactions extracted from the image.'''
        # Open image file
        image = Image.open(image_path).convert('RGB')
        
        # Pass image as input to the model
        coref_results = model.extract_molecule_corefs_from_figures([image])
        return coref_results

    coref_results = get_multi_molecular(image_path)

    def update_symbols_and_corefs(gpt_outputs, coref_results):
        results = []
        for item1, item2 in zip(gpt_outputs, coref_results):
            orig_bboxes = item2.get('bboxes', [])
            orig_corefs = item2.get('corefs', [])
            # 1. Construct new bboxes (prefer exact bbox template, fall back to best-IoU on drift)
            coord2idx = {tuple(bb['bbox']): i for i, bb in enumerate(orig_bboxes)}
            new_bboxes = []
            # Track which new bbox indices each original template expanded into,
            # so corefs can be rebuilt by index.
            orig2new = {}
            for bb1 in item1.get('bboxes', []):
                coord = tuple(bb1['bbox'])
                if coord in coord2idx:
                    tmpl_idx = coord2idx[coord]
                else:
                    tmpl_idx = _ga_best_iou_match_idx(bb1.get('bbox'), orig_bboxes)
                    if tmpl_idx is None:
                        print(f"WARNING [mol-agent]: bbox {coord} not matched to any original template, skipping it.")
                        continue
                bb_template = orig_bboxes[tmpl_idx]
                bb_new = copy.deepcopy(bb_template)
                if 'symbols' in bb1:
                    bb_new['symbols'] = bb1['symbols']
                    if 'atoms' in bb_new:
                        for atom, sym in zip(bb_new['atoms'], bb1['symbols']):
                            atom['atom_symbol'] = sym
                if 'text' in bb1:
                    bb_new['text'] = bb1['text']
                if 'sub_text' in bb1:
                    bb_new['sub_text'] = bb1['sub_text']
                bb_new['bbox'] = bb1['bbox']
                orig2new.setdefault(tmpl_idx, []).append(len(new_bboxes))
                new_bboxes.append(bb_new)

            # 2. Build corefs (rebuild via original-index -> new-index map; skip gracefully on dropped boxes)
            new_corefs = []
            for group in orig_corefs:
                # Assume group = [mol_idx, idt_idx] or [mol_idx1, mol_idx2, ..., idt_idx]
                label_idx = group[-1]
                label_new_list = orig2new.get(label_idx, [])
                if not label_new_list:
                    continue
                new_label_idx = label_new_list[-1]  # label has only one
                # All expanded new indices of mols
                for mol_idx in group[:-1]:
                    for new_mol_idx in orig2new.get(mol_idx, []):
                        new_corefs.append([new_mol_idx, new_label_idx])
            # 3. Assemble structure
            new_item = copy.deepcopy(item2)
            new_item['bboxes'] = new_bboxes
            new_item['corefs'] = new_corefs
            results.append(new_item)
        return results

    input2_updated = update_symbols_and_corefs(gpt_output, coref_results)

    def update_smiles_and_molfile(input_data, conversion_function):
        """
        Use updated symbols, coords, and edges to call `conversion_function` to generate new smiles and molfile,
        and replace them in the original data structure.
        
        Parameters:
        - input_data: nested data structure containing bboxes
        - conversion_function: function accepting coords, symbols, edges and returning (new_smiles, new_molfile, _)
        
        Returns:
        - updated data structure
        """
        for item in input_data:
            for bbox in item.get('bboxes', []):
                # Check whether required keys exist
                if all(key in bbox for key in ['coords', 'symbols', 'edges']):
                    coords = bbox['coords']
                    symbols = bbox['symbols']
                    edges = bbox['edges']
                    
                    # Call conversion function to generate new smiles and molfile
                    new_smiles, new_molfile, _ = conversion_function(coords, symbols, edges)
            
                    # Replace old 'smiles' and 'molfile'
                    bbox['smiles'] = new_smiles
                    bbox['molfile'] = new_molfile

        return input_data

    updated_data = update_smiles_and_molfile(input2_updated, _convert_graph_to_smiles)
    updated_data = _patch_to_mol(updated_data)
    print(f"mol_agent_output:{updated_data}")

    return updated_data
