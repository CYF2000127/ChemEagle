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
import llm_client as llm
from chemietoolkit.mol_edit_plan import SCHEMA as _PLAN_SCHEMA, PlanError, catalog as _plan_catalog, process as _plan_process
from chemietoolkit.mol_edit_plan.annotate import boxed_image_base64 as _boxed_image_base64
from chemietoolkit.mol_edit_plan.postprocess import (add_counter_ion_node as _add_counter_ion_node, tidy_isolated_atoms as _tidy_isolated_atoms,
                                                     repair_ring_bonds as _repair_ring_bonds, drop_dirt_atoms as _drop_dirt_atoms)
import os
import copy
import re
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
            if error_code in (502, 503) or 'overloaded' in error_message.lower() or '503' in error_message or 'Bad Gateway' in error_message:
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

def get_multi_molecular_text_to_correct_azure(image_path: str) -> list:
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

# Graph-level keys the LLM never needs to see; the merge step reads them from
# the full result instead.
_LLM_HIDDEN_KEYS = ("coords", "edges", "molfile", "atoms", "bonds", "category_id", "score", "corefs")


_vision_cache = {}


# Structures the figure draws next to a printed compound label (5a, NHC C, precat. A1). The molecule
# agent links a [Mol] box to its [Idt] box through `corefs`; conditions that name such a label later
# take the drawn structure rather than a name lookup (chemietoolkit.helper.attach_drawn_labelled_structures).
_label_structures = {}


def register_label_structures(image_path, mol_result):
    """Remember (label -> drawn SMILES) for this image (chemietoolkit.helper.drawn_labels_from_mol_result)."""
    from chemietoolkit.helper import drawn_labels_from_mol_result
    found = drawn_labels_from_mol_result(mol_result)
    if found:
        _label_structures.setdefault(image_path, {}).update(found)
    return found


def label_structures(image_path):
    """The (label -> drawn SMILES) map registered for image_path (empty if the agent has not run)."""
    return dict(_label_structures.get(image_path) or {})


def extract_molecule_corefs(image_path: str) -> list:
    '''One MolDetector + Image2Graph + OCR pass over the image. The full graph
    (coords, edges, atoms, bonds, molfile) is kept so the merge step can reuse
    it instead of running the vision models a second time. The pristine result
    is cached per image path (the vision models are deterministic); callers get
    a deep copy, and pristine_vision_result() exposes the untouched boxes for
    cross-checks against the reaction agent.'''
    if image_path not in _vision_cache:
        image = Image.open(image_path).convert('RGB')
        _vision_cache[image_path] = _repair_vision_graphs(model.extract_molecule_corefs_from_figures([image]))
    return copy.deepcopy(_vision_cache[image_path])


def _repair_vision_graphs(result):
    """Deterministic graph repairs on the pristine vision result, before anything reads it: drawing dirt read
    as a loose atom (postprocess.drop_dirt_atoms: a speck that becomes an extra "*." fragment), misplaced ring
    double bonds (postprocess.repair_ring_bonds: a pyrazole read as *C1=NN(*)=CC1) and the ketene patch
    (helper._patch_to_mol: a ketene that lost its central carbon), so the plan catalog, the donor boxes of the
    reaction agent and the final output all see the same corrected graph."""
    for item in result or []:
        for box in item.get('bboxes', []) or []:
            if not all(k in box for k in ('coords', 'symbols', 'edges')):
                continue
            dirt = _drop_dirt_atoms(box)
            changed = _repair_ring_bonds(box)
            if not (changed or dirt):
                continue
            old = box.get('smiles')
            try:
                box['smiles'], box['molfile'], _ = _convert_graph_to_smiles(box['coords'], box['symbols'], box['edges'])
            except Exception as exc:
                print(f"[repair] ring bonds: regeneration failed ({type(exc).__name__}: {exc})")
                continue
            print(f"[repair] {'dirt ' + str(dirt) + ' ' if dirt else ''}{'ring bonds ' + str(changed) if changed else ''}: {old!r} -> {box['smiles']!r}")
    _patch_to_mol(result)
    return result


def pristine_vision_result(image_path: str):
    """The cached vision result for image_path (None if the pass has not run)."""
    return _vision_cache.get(image_path)


_SYMBOL_OPS = {'ocr': 'corrected_symbol', 'atom_ocr': 'corrected_symbol', 'define': 'value', 'define_composite': 'value'}


def corrected_vision_boxes(image_path: str, mol_result=None) -> list:
    """Deep copy of the pristine vision boxes with the edit plan's symbol-level corrections
    (label OCR, atom corrections, definitions) replayed and SMILES regenerated: the generic
    templates as the plan read them, before any expansion into variants. Without a plan audit
    (free-form mode, rejected plan) this is the pristine result. Used as donor graphs when the
    reaction agent's graph for the same box is invalid."""
    pristine = _vision_cache.get(image_path)
    if not pristine:
        return []
    item = copy.deepcopy(pristine[0])
    boxes = item.get('bboxes', [])
    audit = ((mol_result or [{}])[0].get('edit_plan') or {}).get('audit') or []
    touched = set()
    for op in audit:
        if op.get('operation') == 'counter_ion' and op.get('source_bbox_index') is not None and op.get('token'):
            i = op['source_bbox_index']
            if i < len(boxes) and 'symbols' in boxes[i]:
                _add_counter_ion_node(boxes[i], op['token'])
                touched.add(i)
            continue
        field = _SYMBOL_OPS.get(op.get('operation'))
        i, j = op.get('source_bbox_index'), op.get('symbol_index')
        if field is None or i is None or j is None or field not in op:
            continue
        if i < len(boxes) and 'symbols' in boxes[i] and j < len(boxes[i]['symbols']):
            boxes[i]['symbols'][j] = op[field]
            touched.add(i)
    if touched:
        _regenerate_smiles([{'bboxes': [boxes[i] for i in sorted(touched)]}])
    return boxes


def get_multi_molecular_text_to_correct_withatoms(image_path: str, coref_results: Optional[list] = None) -> list:
    '''Tool view of the molecule coref result for the LLM: bbox, category,
    smiles, symbols and text only.

    Pass ``coref_results`` to reuse a pass that already ran. Returns the list
    itself rather than a JSON string, so the caller's json.dumps encodes it
    once instead of embedding an escaped JSON string inside JSON.'''
    if coref_results is None:
        coref_results = extract_molecule_corefs(image_path)
    view = copy.deepcopy(coref_results)
    for item in view:
        for bbox in item.get("bboxes", []):
            for key in _LLM_HIDDEN_KEYS:
                bbox.pop(key, None)
    return view


def _coerce_mol_agent_output(parsed):
    '''Normalise the second-round LLM reply to one dict carrying "bboxes".

    The prompt shows the reply as a dict, but a model may wrap it in a list or
    nest it under a single key such as "output". Accept those shapes and reject
    anything else with a clear error instead of an AttributeError deeper in
    the merge step.'''
    node = parsed
    for _ in range(3):
        if isinstance(node, dict) and isinstance(node.get('bboxes'), list):
            return node
        if isinstance(node, list):
            dicts = [x for x in node if isinstance(x, dict)]
            with_bboxes = [x for x in dicts if isinstance(x.get('bboxes'), list)]
            if with_bboxes:
                return with_bboxes[0]
            if len(dicts) == 1:
                node = dicts[0]
                continue
            break
        if isinstance(node, dict):
            # Descend only into a single dict-like value; a list of strings
            # (the extracted R-group equations) is not a candidate.
            nested = [v for v in node.values()
                      if isinstance(v, dict)
                      or (isinstance(v, list) and any(isinstance(x, dict) for x in v))]
            if len(nested) == 1:
                node = nested[0]
                continue
        break
    keys = list(parsed.keys()) if isinstance(parsed, dict) else 'n/a'
    raise ValueError(
        "molecular agent reply carries no 'bboxes' list at the top level "
        f"(got {type(parsed).__name__}, keys: {keys})")


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
    register_label_structures(image_path, updated_data)
    print(f"mol_agent_output:{updated_data}")

    return updated_data


def process_reaction_image_with_multiple_products_and_text_correctmultiR_azure(image_path: str) -> dict:
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
    
    # Step 1: Tool mapping table. The tool runs MolDetector + Image2Graph once
    # and keeps the full graph in `full_corefs`, so the merge step below reuses
    # it instead of running the vision models a second time.
    full_corefs = {}

    def _withatoms_tool(path: str) -> list:
        if 'result' not in full_corefs:
            full_corefs['result'] = extract_molecule_corefs(path)
        return get_multi_molecular_text_to_correct_withatoms(path, coref_results=full_corefs['result'])

    TOOL_MAP = {
        'get_multi_molecular_text_to_correct_withatoms': _withatoms_tool,
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
    gpt_output = [_coerce_mol_agent_output(json.loads(response.choices[0].message.content))]
    print(f"gpt_output_mol:{gpt_output}")

    # Reuse the pass the tool already ran; the vision models run here only if
    # the model never called the tool.
    if 'result' not in full_corefs:
        full_corefs['result'] = extract_molecule_corefs(image_path)
    coref_results = full_corefs['result']


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
    register_label_structures(image_path, updated_data)
    print(f"mol_agent_output:{updated_data}")

    return updated_data


############################### ChemEagle (any OpenAI-compatible endpoint)
def process_reaction_image_with_multiple_products_and_text_correctmultiR(
    image_path: str,
    *,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    coref_results: Optional[list] = None,
    artifacts: Optional[dict] = None,
) -> dict:
    """
    Aligned with process_reaction_image_with_multiple_products_and_text_correctmultiR workflow, but uses the OpenAI-compatible endpoint configured in llm.get_client().

    Args:
        image_path: image file path.
        model_name: model id (default: llm.resolve_model()).
        base_url: API base URL (default: llm.resolve_base_url()).
        api_key: API key (default: API_KEY env).
        coref_results: a previous extract_molecule_corefs() result to reuse (skips the vision models).
        artifacts: optional dict that receives raw_reply, llm_parsed and usage for evaluation.

    Returns:
        dict: organized reaction data, including reactants, products, and reaction templates.
    """
    model_name = llm.resolve_model(model_name)
    base_url = llm.resolve_base_url(base_url)
    api_key = llm.resolve_key(api_key)
    _mk = llm.model_kwargs(model_name)

    client = llm.get_client(api_key=api_key, base_url=base_url)

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
        **_mk,
        #response_format={'type': 'json_object'},  # response_format is applied on the final call only
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    # Step 1: Tool mapping table. The tool runs MolDetector + Image2Graph once
    # and keeps the full graph in `full_corefs`, so the merge step below reuses
    # it instead of running the vision models a second time.
    full_corefs = {'result': coref_results} if coref_results is not None else {}

    def _withatoms_tool(path: str) -> list:
        if 'result' not in full_corefs:
            full_corefs['result'] = extract_molecule_corefs(path)
        return get_multi_molecular_text_to_correct_withatoms(path, coref_results=full_corefs['result'])

    TOOL_MAP = {
        'get_multi_molecular_text_to_correct_withatoms': _withatoms_tool,
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
    response, _ = llm.final_json_call(
        client, completion_payload["model"], completion_payload["messages"], _mk,
        tool_map=TOOL_MAP, tool_arg=image_path, retry=retry_api_call)

    # Get GPT-generated result
    raw_content = response.choices[0].message.content

    try:
        gpt_output = [_coerce_mol_agent_output(json.loads(raw_content))]
        print(f"DEBUG [agent]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        print(f"ERROR [agent]: Failed to parse JSON from model response")
        print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
        raise json.JSONDecodeError(
            f"Could not parse JSON from model response. Content may not be valid JSON.",
            raw_content, 0
        )
    
    if artifacts is not None:
        artifacts.update(raw_reply=raw_content, llm_parsed=gpt_output[0],
                         usage=response.usage.model_dump() if getattr(response, 'usage', None) else None)
    print(f"gpt_output_mol:{gpt_output}")

    # Reuse the pass the tool already ran; the vision models run here only if
    # the model never called the tool.
    if 'result' not in full_corefs:
        full_corefs['result'] = extract_molecule_corefs(image_path)
    coref_results = full_corefs['result']

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
    register_label_structures(image_path, updated_data)
    print(f"mol_agent_output:{updated_data}")

    return updated_data


############################### ChemEagle: edit-plan variant of the molecular agent
_PLAN_PROMPT_PATH = './prompt/prompt_Mol_Plan.txt'


def _regenerate_smiles(items):
    """Graph2SMILES over every molecule that carries coords/symbols/edges (the
    update_smiles_and_molfile step of correctmultiR, as a module-level helper)."""
    for item in items:
        for bbox in item.get('bboxes', []):
            if all(k in bbox for k in ('coords', 'symbols', 'edges')):
                dropped = _tidy_isolated_atoms(bbox)      # stray "Br" from a label, a counter-ion read twice
                if dropped:
                    print(f"[tidy] dropped isolated atoms {dropped} from a molecule box (bbox {bbox.get('bbox')})")
                new_smiles, new_molfile, _ = _convert_graph_to_smiles(bbox['coords'], bbox['symbols'], bbox['edges'])
                bbox['smiles'] = new_smiles
                bbox['molfile'] = new_molfile
    return items


def _merge_plan_output(plan_out, full_item):
    """Rebuild the full-graph result from the postprocess output.

    Every output molecule/text box names its source box through provenance, so the
    full box (coords, edges, atoms, molfile ...) is copied by index and only the
    edited fields (symbols, text) are overlaid; no bbox matching is needed. Labels
    the plan derived for expanded variants have no source box; when the detector
    missed their text block they get the parent molecule's bbox so downstream code
    that expects numeric boxes keeps working (bbox_provenance records that)."""
    src = full_item['bboxes']
    prov = {row['output_index']: row for row in plan_out['postprocess']['provenance']}
    parent_of = {ti: mi for mi, ti in plan_out['corefs']}
    new_boxes = []
    for oi, ob in enumerate(plan_out['bboxes']):
        row = prov.get(oi)
        if row is not None:
            box = copy.deepcopy(src[row['source_bbox_index']])
            if 'symbols' in ob:
                box['symbols'] = list(ob['symbols'])
                for atom, sym in zip(box.get('atoms', []) or [], ob['symbols']):
                    atom['atom_symbol'] = sym
            if 'text' in ob:
                box['text'] = list(ob['text'])
            if row.get('compound_id'):
                box['compound_id'] = row['compound_id']
                box['variant_id'] = row['variant_id']
            if ob.get('counter_ion'):
                _add_counter_ion_node(box, ob['counter_ion'])     # drawn free anion: one more isolated atom, expanded by Graph2SMILES
                box['counter_ion'] = ob['counter_ion']
            new_boxes.append(box)
            continue
        label = copy.deepcopy(ob)
        if label.get('bbox') is None:
            parent = plan_out['bboxes'][parent_of[oi]]
            label['bbox'] = list(parent['bbox'])
            label['bbox_provenance'] = 'parent_molecule'
        new_boxes.append(label)
    merged = copy.deepcopy(full_item)
    merged['bboxes'] = new_boxes
    merged['corefs'] = [list(c) for c in plan_out['corefs']]
    # provenance rides along: it is what the variants of an expansion group were bound to, and the table agent
    # reads it when a row names a drawn compound ("7": "7a") without printing that compound's own values
    merged['edit_plan'] = {k: plan_out['postprocess'][k] for k in ('status', 'decisions', 'structure_warnings', 'audit', 'provenance')}
    merged['extracted_explicit_rgroup_equations'] = plan_out['extracted_explicit_rgroup_equations(without any reasoning and infer)']
    return merged


def process_reaction_image_with_multiple_products_and_text_correctmultiR_plan(
    image_path: str,
    *,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    coref_results: Optional[list] = None,
    artifacts: Optional[dict] = None,
    fallback_to_freeform: bool = False,
    retry_on_reject: bool = True,
) -> list:
    """Edit-plan variant of process_reaction_image_with_multiple_products_and_text_correctmultiR.

    Same vision pass and same result shape, but the LLM no longer rewrites the
    bboxes array. It receives the image plus a catalog of immutable ids and returns
    a small edit plan (OCR corrections, text corrections, single-valued definitions,
    expansion groups, decisions) under a strict JSON schema; chemietoolkit.mol_edit_plan.process
    applies it deterministically and the full graph is rebuilt by provenance.

    Args:
        image_path: image file path.
        model_name / base_url / api_key: the configured endpoint settings (see llm_client).
        coref_results: a previous extract_molecule_corefs() result to reuse (skips the vision models).
        artifacts: optional dict that receives catalog, plan, raw_reply, usage, postprocess_output.
        retry_on_reject: when the validator rejects the plan, ask the model once more with
            the validator's message appended.
        fallback_to_freeform: if the plan is still rejected, run the free-form correctmultiR
            path (default False: the vision result is returned unchanged and flagged, because the
            free-form rewrite can corrupt the graph).
    """
    model_name = llm.resolve_model(model_name)
    base_url = llm.resolve_base_url(base_url)
    api_key = llm.resolve_key(api_key)
    _mk = llm.model_kwargs(model_name)
    client = llm.get_client(api_key=api_key, base_url=base_url)

    if coref_results is None:
        coref_results = extract_molecule_corefs(image_path)
    full_item = coref_results[0]
    # The catalog and the validator read the full vision result: atom-level rules need the
    # edge matrix (bond counts) and coordinates; only the catalog's own fields reach the prompt.
    cat, _mols, _texts, _atoms = _plan_catalog(full_item)

    with open(_PLAN_PROMPT_PATH, 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    # MOL_PLAN_BOXED_IMAGE=1 (default): the model sees the figure with the catalog's molecule boxes
    # outlined and tagged by number, so "mol_003" is a place in the picture and not a set of
    # coordinates. 0 sends the bare figure (the 2026-09-14 behaviour).
    boxed = os.environ.get('MOL_PLAN_BOXED_IMAGE', '1') != '0'
    if boxed:
        base64_image = _boxed_image_base64(image_path, cat['molecules'], min_side=700)
        user_text = ('Inspect this image and the detector-derived catalog; produce the edit plan. The thin blue boxes '
                     'and numeric tags are a program overlay, not part of the figure: tag N marks the box of catalog '
                     'molecule mol_00N (tag 3 = mol_003), and each tag sits just outside its box.')
    else:
        with open(image_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        user_text = 'Inspect this image and the detector-derived catalog; produce the edit plan.'
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': user_text},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}', 'detail': 'high'}},
            {'type': 'text', 'text': json.dumps(cat, ensure_ascii=False)}]}]
    response = retry_api_call(
        client.chat.completions.create,
        max_retries=5, base_delay=3, backoff_factor=2,
        model=model_name, messages=messages,
        response_format={'type': 'json_schema', 'json_schema': {'name': 'mol_edit_plan', 'strict': True, 'schema': _PLAN_SCHEMA}},
        **_mk)
    raw_content = response.choices[0].message.content
    if artifacts is not None:
        artifacts.update(catalog=cat, raw_reply=raw_content,
                         usage=response.usage.model_dump() if getattr(response, 'usage', None) else None)
    if not raw_content or not raw_content.strip():
        raise ValueError('Model returned empty content instead of an edit plan')
    try:
        plan = json.loads(raw_content)
    except json.JSONDecodeError:
        # diagnostics only (2026-09-13, local vLLM models): the reply is not kept anywhere else
        print(f"ERROR [mol-agent]: unparsable plan reply; finish_reason={getattr(response.choices[0], 'finish_reason', None)} "
              f"usage={getattr(response, 'usage', None)} len={len(raw_content)}", flush=True)
        print("Raw plan content (last 2000 chars):", flush=True)
        print(raw_content[-2000:], flush=True)
        raise
    if artifacts is not None:
        artifacts['plan'] = plan
    print(f"DEBUG [mol-agent]: {len(plan.get('ocr_corrections', []))} OCR, {len(plan.get('text_corrections', []))} text, "
          f"{len(plan.get('definitions', []))} definitions, {len(plan.get('groups', []))} groups, "
          f"{sum(len(g.get('variants', [])) for g in plan.get('groups', []))} variants, {len(plan.get('decisions', []))} decisions")
    plan_out, last_error = None, None
    for attempt in range(2 if retry_on_reject else 1):
        try:
            plan_out = _plan_process(full_item, plan)
            break
        except PlanError as exc:
            last_error = str(exc)
            print(f"WARNING [mol-agent]: plan rejected by the validator (attempt {attempt + 1}): {exc}")
            if artifacts is not None:
                artifacts.setdefault('plan_errors', []).append(last_error)
            if attempt == 0 and retry_on_reject:
                messages = messages + [
                    {'role': 'assistant', 'content': raw_content},
                    {'role': 'user', 'content': f"The plan was rejected by the validator: {last_error}. "
                                                "Return a corrected plan that satisfies the schema and the rules; keep every other entry."}]
                response = retry_api_call(
                    client.chat.completions.create,
                    max_retries=5, base_delay=3, backoff_factor=2,
                    model=model_name, messages=messages,
                    response_format={'type': 'json_schema', 'json_schema': {'name': 'mol_edit_plan', 'strict': True, 'schema': _PLAN_SCHEMA}},
                    **_mk)
                raw_content = response.choices[0].message.content or ''
                if not raw_content.strip():
                    break
                plan = json.loads(raw_content)
                if artifacts is not None:
                    artifacts['plan_retry'] = plan
                    artifacts['raw_reply_retry'] = raw_content
    if plan_out is None and plan is not None and os.environ.get('MOL_PLAN_LENIENT', '1') != '0':
        # MOL_PLAN_LENIENT=1 (default): keep what the model got right. The entries that fail
        # validation are dropped and audited; the rest of the plan is applied.
        from chemietoolkit.mol_edit_plan.postprocess import process_lenient
        try:
            plan_out = process_lenient(full_item, plan)
            post = plan_out['postprocess']
            print(f"WARNING [mol-agent]: plan applied leniently: {len(post.get('dropped_entries', []))} entr{'y' if len(post.get('dropped_entries', [])) == 1 else 'ies'} dropped, "
                  f"{len(post.get('added_decisions', []))} decision(s) added; first drop: {str(post.get('dropped_entries', [{}])[0].get('reason'))[:120] if post.get('dropped_entries') else '-'}")
            if artifacts is not None:
                artifacts['plan_lenient'] = {'dropped': post.get('dropped_entries'), 'added': post.get('added_decisions')}
        except PlanError as exc:
            last_error = f"{last_error}; lenient: {exc}"
            print(f"WARNING [mol-agent]: lenient application failed too: {exc}")
    if plan_out is None:
        if artifacts is not None:
            artifacts['plan_error'] = last_error
        if fallback_to_freeform:
            print("WARNING [mol-agent]: falling back to the free-form correctmultiR path")
            return process_reaction_image_with_multiple_products_and_text_correctmultiR(
                image_path, model_name=model_name, base_url=base_url, api_key=api_key,
                coref_results=coref_results, artifacts=artifacts)
        print("WARNING [mol-agent]: plan still rejected; returning the vision result unchanged (needs review)")
        untouched = copy.deepcopy(full_item)
        untouched['edit_plan'] = {'status': 'plan_rejected', 'error': last_error, 'decisions': [], 'structure_warnings': [], 'audit': []}
        return _patch_to_mol(_regenerate_smiles([untouched]))
    if artifacts is not None:
        artifacts['postprocess_output'] = plan_out
    print(f"mol_plan_equations:{plan_out['extracted_explicit_rgroup_equations(without any reasoning and infer)']} status={plan_out['postprocess']['status']}")
    updated_data = _regenerate_smiles([_merge_plan_output(plan_out, full_item)])
    updated_data = _patch_to_mol(updated_data)
    register_label_structures(image_path, updated_data)
    print(f"mol_agent_output:{updated_data}")
    return updated_data


def molecular_agent(image_path: str, **kwargs) -> list:
    """Molecular recognition agent: the model returns a small edit plan that is
    applied deterministically (correctmultiR_plan).

    The older free-form variant, where the model rewrites the whole bboxes array
    (process_reaction_image_with_multiple_products_and_text_correctmultiR), is
    kept in this file but no longer wired into the pipeline."""
    return process_reaction_image_with_multiple_products_and_text_correctmultiR_plan(image_path, **kwargs)
