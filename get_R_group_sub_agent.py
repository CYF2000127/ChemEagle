import sys
import torch
import json
from chemietoolkit import ChemIEToolkit,utils
import cv2
import numpy as np
from PIL import Image
import json
from get_molecular_agent import process_reaction_image_with_multiple_products_and_text_correctR, process_reaction_image_with_multiple_products_and_text_correctmultiR, process_reaction_image_with_multiple_products_and_text_correctmultiR_OS
from get_reaction_agent import get_reaction_withatoms_correctR, get_reaction_withatoms_correctR_OS, get_reaction_con, get_reaction_con_OS
import sys
from rxnim import RxnIM
import json
import base64
import torch
import json
from PIL import Image
import numpy as np
from openai import AzureOpenAI,  OpenAI
from typing import Optional
import copy
from molnextr.chemistry import _convert_graph_to_smiles 
import os
import io
import re
import time
from openai import InternalServerError, RateLimitError, APIError



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChemIEToolkit(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
ckpt_path = "./rxn.ckpt"
model1 = RxnIM(ckpt_path, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")


def normalize_product_variant_output(data: dict) -> dict:
    """
    Convert output format of _product_variant_R_group series functions to a standardized format.
    
    Input format:
    {
        'reaction_template': {
            'reactants': ['SMILES1', 'SMILES2'],
            'products': ['SMILES3']
        },
        'reactions': {
            '20': {'reactants': [...], 'products': [...]},
            '25': {'reactants': [...], 'products': [...]},
            ...
        },
        'original_molecule_list': {...}
    }
    
    Output format:
    {
        'reactions': [
            {
                'reaction_id': '0_1',
                'note': 'reaction template',
                'reactants': [{'smiles': 'SMILES1'}, {'smiles': 'SMILES2'}],
                'conditions': [],
                'products': [{'smiles': 'SMILES3', 'label': '3'}]
            },
            {
                'reaction_id': '1_1',
                'reactants': [{'smiles': '...'}, {'smiles': '...'}],
                'conditions': ['8 h, 87% yield'],
                'products': [{'smiles': '...', 'label': '20'}]
            },
            ...
        ],
        'original_molecule_list': {...}  # keep unchanged
    }
    """
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary")
    
    normalized_reactions = []
    original_molecule_list = data.get('original_molecule_list', {})
    
    # 1. Process reaction_template (first reaction, reaction_id = '0_1')
    if 'reaction_template' in data:
        template = data['reaction_template']
        template_reactants = template.get('reactants', [])
        template_products = template.get('products', [])
        
        # Try to find product template label from original_molecule_list
        template_product_labels = []
        if template_products:
            # Find corresponding label for each product
            for template_product_smiles in template_products:
                template_product_label = None
                # Find matching SMILES in original_molecule_list and locate template-related entries
                if template_product_smiles in original_molecule_list:
                    info = original_molecule_list[template_product_smiles]
                    if isinstance(info, list) and len(info) > 0:
                        # Check whether it is template-related (info may contain 'reactant template' or 'product template')
                        info_str = ' '.join(str(item).lower() for item in info)
                        if 'template' in info_str:
                            # Get label from info[0]
                            template_product_label = str(info[0]) if info[0] else None
                template_product_labels.append(template_product_label)
        
        normalized_reactions.append({
            'reaction_id': '0_1',
            'note': 'reaction template',
            'reactants': [{'smiles': smiles} for smiles in template_reactants],
            'conditions': [],
            'products': [{'smiles': smiles, 'label': label} if label else {'smiles': smiles}
                        for smiles, label in zip(template_products, template_product_labels)]
        })
    
    # 1.5 Collect molecules without explicit reactant/product roles into reaction template conditions
    #     Reactant/product role keywords:
    reactant_product_roles = {'product', 'reactant', 'reactant template', 'product template'}
    #     Condition role keywords (also put into conditions):
    condition_roles = {'condition', 'conditions'}
    #     All role words to skip when extracting labels:
    skip_words = reactant_product_roles | condition_roles

    # First collect all SMILES already present in reactants/products for deduplication
    all_rxn_smiles = set()
    if 'reaction_template' in data:
        template = data['reaction_template']
        for s in template.get('reactants', []):
            all_rxn_smiles.add(s)
        for s in template.get('products', []):
            all_rxn_smiles.add(s)
    if 'reactions' in data:
        for rxn_data in data['reactions'].values():
            for s in rxn_data.get('reactants', []):
                all_rxn_smiles.add(s)
            for s in rxn_data.get('products', []):
                all_rxn_smiles.add(s)

    unassigned_conditions = []
    for smiles, info in original_molecule_list.items():
        if not isinstance(info, list):
            continue
        # Skip molecules already present in reactants/products
        if smiles in all_rxn_smiles:
            continue
        # Combine all items in info into a lowercase set and check roles
        info_lower = [str(item).lower().strip() for item in info]
        has_reactant_product_role = any(role in info_lower for role in reactant_product_roles)
        if has_reactant_product_role:
            continue
        # No reactant/product role -> put into conditions (including explicitly marked condition/conditions and those with no role)
        label_parts = []
        for item in info:
            if isinstance(item, str) and not item.startswith('bbox_id=') and not item.startswith('id='):
                if item.lower().strip() not in skip_words:
                    label_parts.append(item)
        label = ', '.join(label_parts) if label_parts else ''
        cond_entry = {'role': 'reagent', 'smiles': smiles}
        if label:
            cond_entry['label'] = f'{label} (the label maybe wrong, please check the image again)'
        unassigned_conditions.append(cond_entry)
    
    # Inject unassigned molecules into reaction template conditions
    if unassigned_conditions and normalized_reactions:
        normalized_reactions[0]['conditions'].extend(unassigned_conditions)
    
    # 2. Process reactions dictionary (numbering starts from '1_1')
    if 'reactions' in data:
        reactions_dict = data['reactions']
        # Sort by key to ensure consistent order
        sorted_reaction_keys = sorted(reactions_dict.keys())
        
        for idx, reaction_key in enumerate(sorted_reaction_keys, start=1):
            reaction_data = reactions_dict[reaction_key]
            reaction_reactants = reaction_data.get('reactants', [])
            reaction_products = reaction_data.get('products', [])
            
            # Extract conditions: match by product SMILES and label from original_molecule_list
            conditions = []
            for product_smiles in reaction_products:
                # Find matching entries in original_molecule_list
                if product_smiles in original_molecule_list:
                    info = original_molecule_list[product_smiles]
                    if isinstance(info, list) and len(info) > 0:
                        # info format: ['20', '8 h, 87% yield', 'product', 'bbox_id=14', 'id=15']
                        # Need to find entry with matching label (info[0] is label)
                        info_label = str(info[0]) if info[0] else None
                        if info_label == str(reaction_key):
                            # Extract condition info (skip label(index 0), 'product'/'reactant'/'template', bbox_id=, id=)
                            for item in info[1:]:
                                if item and isinstance(item, str):
                                    # Skip fixed keywords and items starting with bbox_id= or id=
                                    if item not in ['product', 'reactant', 'template'] and not item.startswith('bbox_id=') and not item.startswith('id='):
                                        if item not in conditions:  # Avoid duplicates
                                            conditions.append(item)
            
            # Build standardized format
            normalized_reaction = {
                'reaction_id': f'{idx}_1',
                'reactants': [{'smiles': smiles} for smiles in reaction_reactants],
                'conditions': conditions if conditions else [],
                'products': [{'smiles': smiles, 'label': reaction_key} for smiles in reaction_products]
            }
            normalized_reactions.append(normalized_reaction)
    
    return {
        'reactions': normalized_reactions,
        'original_molecule_list': original_molecule_list
    }


def retry_api_call(func, max_retries=3, base_delay=2, backoff_factor=2, *args, **kwargs):
    """
    Generic API call retry function with exponential backoff support.
    
    Args:
        func: function to call
        max_retries: maximum number of retries
        base_delay: base delay time (seconds)
        backoff_factor: backoff factor (retry delay = base_delay * backoff_factor^attempt)
        *args, **kwargs: parameters passed to func
    
    Returns:
        return value of func
    
    Raises:
        exception from the final attempt
    """
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



def _compensate_missing_molecules(gpt_output: dict, results: list, tool_name: str) -> dict:
    """
    Compensation mechanism: supplement molecules missed by GPT in tool output into gpt_output.

    Args:
        gpt_output: molecule dictionary generated by GPT, key is SMILES
        results: tool call result list
        tool_name: name of molecule recognition tool

    Returns:
        supplemented gpt_output
    """
    # Extract output of molecule recognition tool from results
    tool_molecules = None
    for r in results:
        content_dict = json.loads(r['content'])
        if tool_name in content_dict:
            tool_molecules = content_dict[tool_name]
            break

    if not tool_molecules or not isinstance(tool_molecules, list):
        return gpt_output

    # Find the existing maximum id in gpt_output
    max_id = 0
    for smiles, info in gpt_output.items():
        if isinstance(info, list):
            for item in info:
                if isinstance(item, str) and item.startswith('id='):
                    try:
                        id_val = int(item.split('=')[1])
                        max_id = max(max_id, id_val)
                    except ValueError:
                        pass

    # Check and supplement missed molecules
    added_count = 0
    for mol in tool_molecules:
        smiles = mol.get('smiles', '')
        if smiles and smiles not in gpt_output:
            max_id += 1
            texts = mol.get('texts', mol.get('text', []))
            if isinstance(texts, str):
                texts = [texts]
            texts_str = ', '.join(str(t) for t in texts) if texts else ''
            bbox_id = mol.get('bbox_id', '')
            entry = [texts_str, f'bbox_id={bbox_id}', f'id={max_id}']
            gpt_output[smiles] = entry
            added_count += 1
            print(f"[Compensation] Added missing molecule: {smiles} -> {entry}")

    if added_count > 0:
        print(f"[Compensation] Total {added_count} missing molecule(s) compensated.")

    return gpt_output

def draw_mol_bboxes(image_path, coref_results, output_path=None):
    """
    Draws bounding boxes on the original image for each molecule in coref_results.
    
    Args:
        image_path (str): Path to the original image.
        coref_results (list): The coreference results data structure.
        output_path (str, optional): Path to save the annotated image. If None, returns the image array.
    
    Returns:
        np.ndarray: The annotated image if output_path is None, else None (saves file).
    """
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist.")
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    height, width = img.shape[:2]

    # Handle the structure of coref_results
    # It is a list of dicts. We assume the first dict corresponds to the image if it's a list.
    if isinstance(coref_results, list) and len(coref_results) > 0:
        data = coref_results[0]
    elif isinstance(coref_results, dict):
        data = coref_results
    else:
        print("Error: Invalid coref_results format")
        return None

    bboxes = data.get('bboxes', [])
    
    # Iterate through all bounding boxes
    for item in bboxes:
        # Check if category is [Mol]
        if item.get('category') == '[Mol]':
            bbox = item.get('bbox')
            bbox_id = item.get('bbox_id')
            
            if bbox:
                # Bbox is [x1, y1, x2, y2] normalized
                # Convert to pixel coordinates
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                
                # Draw rectangle (Green, thickness 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw bbox_id
                if bbox_id is not None:
                    label = str(bbox_id)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    
                    # Calculate text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Ensure text is within image bounds
                    text_x = x1
                    text_y = y1 - 5
                    
                    # If box is at the top, draw text inside or below
                    if text_y < text_height:
                        text_y = y1 + text_height + 5
                        
                    # Draw background rectangle for text
                    # cv2.rectangle(img, (text_x, text_y - text_height - 2), (text_x + text_width, text_y + baseline), (0, 255, 0), -1)
                    # Use a small filled box for the ID
                    cv2.rectangle(img, (x1, y1), (x1 + text_width + 4, y1 + text_height + 10), (255, 255, 255), -1)
                    
                    # Draw text (Black)
                    cv2.putText(img, label, (x1 + 2, y1 + text_height + 5), font, font_scale, (0, 0, 0), thickness)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to {output_path}")
        return None
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_json_from_text_with_reasoning(text):
    """
    Extract JSON object from text containing reasoning process.
    Supports processing output from thinking models where large reasoning text may appear before final JSON.
    
    Args:
        text: text containing JSON, possibly with reasoning process
        
    Returns:
        dict: parsed JSON object, return None on failure
    """
    # Method 1: try parsing the whole text directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Method 2: find JSON after </think> or similar marker
    # Handle markers possibly used by thinking models
    markers = [
        r'</think>',
        r'</thinking>',
        r'</reasoning>',
        r'</think>',
        r'```json',
        r'```',
    ]
    
    for marker in markers:
        pattern = f'{marker}\\s*(.*?)(?:```|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                continue
    
    # Method 3: search backward from end for complete JSON object
    # Find last { position, then try matching complete JSON
    last_brace_start = text.rfind('{')
    if last_brace_start != -1:
        # Start from last { and try finding matching }
        brace_count = 0
        json_end = -1
        for i in range(last_brace_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end != -1:
            json_content = text[last_brace_start:json_end]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
    
    # Method 4: find content between first { and last } (simple method)
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_content = text[first_brace:last_brace + 1]
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass
    
    # Method 5: find JSON containing "reactions" key (for specific format)
    reactions_pattern = r'\{[^{}]*"reactions"[^{}]*\[.*?\].*?\}'
    match = re.search(reactions_pattern, text, re.DOTALL)
    if match:
        # Expand matching range to find complete JSON object
        start = match.start()
        # Find { backward and matching } forward
        brace_count = 0
        json_start = start
        json_end = -1
        
        # Search backward for starting {
        for i in range(start, -1, -1):
            if text[i] == '}':
                brace_count += 1
            elif text[i] == '{':
                brace_count -= 1
                if brace_count == 0:
                    json_start = i
                    break
        
        if json_start != -1:
            brace_count = 0
            for i in range(json_start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end != -1:
                json_content = text[json_start:json_end]
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass
    
    return None


def parse_coref_data_with_fallback(data):
    bboxes = data["bboxes"]
    corefs = data["corefs"]
    paired_indices = set()

    # Process coref-paired items first
    results = []
    for idx1, idx2 in corefs:
        smiles_entry = bboxes[idx1] if "smiles" in bboxes[idx1] else bboxes[idx2]
        text_entry = bboxes[idx2] if "text" in bboxes[idx2] else bboxes[idx1]

        smiles = smiles_entry.get("smiles", "")
        #bbox= smiles_entry.get("bbox", ())
        bbox_id = smiles_entry.get("bbox_id", "")
        
        # If smiles_entry has sub_text, use it directly; otherwise use text from text_entry
        if "sub_text" in smiles_entry:
            result_item = {
                "smiles": smiles,
                "text": smiles_entry["sub_text"],
                #"bbox": bbox,
                "bbox_id": bbox_id
            }
        else:
            texts = text_entry.get("text", [])
            result_item = {
                "smiles": smiles,
                "texts": texts,
                #"bbox": bbox,
                "bbox_id": bbox_id
            }
        
        results.append(result_item)

        # Record which SMILES have been paired
        paired_indices.add(idx1)
        paired_indices.add(idx2)

    # Process unpaired SMILES (supplement them)
    for idx, entry in enumerate(bboxes):
        if "smiles" in entry and idx not in paired_indices:
            # If entry has sub_text, use it directly; otherwise use default prompt text
            if "sub_text" in entry:
                result_item = {
                    "smiles": entry["smiles"],
                    "text": entry["sub_text"],
                    #"bbox": entry["bbox"],
                    "bbox_id": entry.get("bbox_id", ""),
                }
            else:
                result_item = {
                    "smiles": entry["smiles"],
                    "texts": ["There is no label or failed to detect, please recheck the image again"],
                    #"bbox": entry["bbox"],
                    "bbox_id": entry.get("bbox_id", ""),
                }
            results.append(result_item)

    return results

def parse_coref_data_with_fallback_with_box(data):
    bboxes = data["bboxes"]
    corefs = data["corefs"]
    paired_indices = set()

    # Process coref-paired items first
    results = []
    for idx1, idx2 in corefs:
        smiles_entry = bboxes[idx1] if "smiles" in bboxes[idx1] else bboxes[idx2]
        text_entry = bboxes[idx2] if "text" in bboxes[idx2] else bboxes[idx1]

        smiles = smiles_entry.get("smiles", "")
        bboxes = smiles_entry.get("bbox", [])
        texts = text_entry.get("text", [])

        results.append({
            "smiles": smiles,
            "texts": texts,
            "bbox": bboxes
        })

        # Record which SMILES have been paired
        paired_indices.add(idx1)
        paired_indices.add(idx2)

    # Process unpaired SMILES (supplement them)
    for idx, entry in enumerate(bboxes):
        if "smiles" in entry and idx not in paired_indices:
            results.append({
                "smiles": entry["smiles"],
                "texts": ["There is no label or failed to detect, please recheck the image again"],
                "bbox": entry["bbox"],
            })

    return results





############################### MOl
_process_multi_molecular_cache = {}

def get_cached_multi_molecular(image_path: str):
    """
    Only truly call once for the same image_path
    process_reaction_image_with_multiple_products_and_text_correctR
    and cache the result.
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    if image_path not in _process_multi_molecular_cache:
        ##print(f"[get_cached_multi_molecular] Processing image: {image_path}")
        _process_multi_molecular_cache[image_path] = (
            process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path)
            ################################model.extract_molecule_corefs_from_figures([image])#############################################################################################
            )
        ##print(f"original output: {model.extract_molecule_corefs_from_figures([image])}")
    return _process_multi_molecular_cache[image_path]


def get_multi_molecular_text_to_correct(image_path: str) -> list:
    """
    Tool registered for GPT-4o. Internally no longer directly calls second-level Agent,
    but reuses cached results.
    """
    coref_results = copy.deepcopy(get_cached_multi_molecular(image_path))

    # Delete fields not intended for LLM return as needed
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in [
                "category", "molfile", "symbols",
                "atoms", "bonds", "category_id", "score", "corefs",
                "coords", "edges"
            ]:
                bbox.pop(key, None)

    # Assume parse_coref_data_with_fallback requires a single dict input
    parsed = parse_coref_data_with_fallback(coref_results[0])
    print(f"[get_multi_molecular_text_to_correct] parsed: {json.dumps(parsed)}")
    return parsed

############################### MOl_OS
_process_multi_molecular_cache = {}

def get_cached_multi_molecular_OS(image_path: str):
    """
    Only truly call once for the same image_path
    process_reaction_image_with_multiple_products_and_text_correctR
    and cache the result.
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    if image_path not in _process_multi_molecular_cache:
        ##print(f"[get_cached_multi_molecular] Processing image: {image_path}")
        _process_multi_molecular_cache[image_path] = (
            process_reaction_image_with_multiple_products_and_text_correctmultiR_OS(image_path)
            #######model.extract_molecule_corefs_from_figures([image])
            )
        ##print(f"original output: {model.extract_molecule_corefs_from_figures([image])}")
    return _process_multi_molecular_cache[image_path]


def get_multi_molecular_text_to_correct_OS(image_path: str) -> list:
    """
    Tool registered for GPT-4o. Internally no longer directly calls second-level Agent,
    but reuses cached results.
    """
    coref_results = copy.deepcopy(get_cached_multi_molecular_OS(image_path))

    # Delete fields not intended for LLM return as needed
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in [
                "category", "molfile", "symbols",
                "atoms", "bonds", "category_id", "score", "corefs",
                "coords", "edges"
            ]:
                bbox.pop(key, None)

    # Assume parse_coref_data_with_fallback requires a single dict input
    parsed = parse_coref_data_with_fallback(coref_results[0])
    print(f"[get_multi_molecular_text_to_correct] parsed: {json.dumps(parsed)}")
    return parsed



def get_multi_molecular_full(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # Open image file
    image = Image.open(image_path).convert('RGB')
    
    # Pass image as input to the model
    coref_results = process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path)
    #coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # Safely remove key

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)
    return parsed

def get_multi_molecular_full_OS(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # Open image file
    image = Image.open(image_path).convert('RGB')
    
    # Pass image as input to the model
    coref_results = process_reaction_image_with_multiple_products_and_text_correctmultiR_OS(image_path)
    #coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # Safely remove key

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)
    return parsed



############################### Rxn
_raw_results_cache = {}

def get_cached_raw_results(image_path: str):
    """
    Call get_reaction_withatoms_correctR once and cache the result,
    then reuse the same raw_results afterward.
    """
    if image_path not in _raw_results_cache:
        #print(f"[get_cached_raw_results] Processing image: {image_path}")
        _raw_results_cache[image_path] = get_reaction_withatoms_correctR(image_path)
        ###############################_raw_results_cache[image_path]= model1.predict_image_file(image_path, molnextr=True, ocr=True)####################################################################
    return _raw_results_cache[image_path]


# ----------------------------------------
# Utility function: build compact output based on raw_pred
# ----------------------------------------
def get_reaction_from_raw(raw_pred: dict) -> dict:
    """
    Returns a structured dictionary of reactions extracted from the raw prediction,
    """
    structured = {}
    for section in ['reactants', 'conditions', 'products']:
        if section in raw_pred:
            structured[section] = []
            for item in raw_pred[section]:
                if section in ('reactants', 'products'):
                    structured[section].append({
                        "smiles": item.get("smiles", ""),
                        "bbox":   item.get("bbox",   [])
                    })
                else:  # conditions
                    structured[section].append({
                        "text":   item.get("text",   []),
                        "bbox":   item.get("bbox",   []),
                        "smiles": item.get("smiles", [])
                    })
    return structured

# ----------------------------------------
# LLM tool: get_reaction
# ----------------------------------------
def get_reaction(image_path: str) -> dict:
    """    
    Returns a structured dictionary of reactions extracted from the image,
    """
    # Reuse cached raw_results
    raw_results = get_cached_raw_results(image_path)
    if not raw_results:
        # No reaction detected: return an empty result instead of an IndexError.
        return {}
    raw_pred = raw_results[0]
    return get_reaction_from_raw(raw_pred)

############################### Rxn_OS

def get_cached_raw_results_OS(image_path: str):
    """
    Call get_reaction_withatoms_correctR once and cache the result,
    then reuse the same raw_results afterward.
    """
    if image_path not in _raw_results_cache:
        #print(f"[get_cached_raw_results] Processing image: {image_path}")
        _raw_results_cache[image_path]= get_reaction_withatoms_correctR_OS(image_path)
        ######_raw_results_cache[image_path]= model1.predict_image_file(image_path, molnextr=True, ocr=True)####################################################################
    return _raw_results_cache[image_path]



def get_reaction_OS(image_path: str) -> dict:
    """    
    Returns a structured dictionary of reactions extracted from the image,
    """
    # Reuse cached raw_results
    raw_results = get_cached_raw_results_OS(image_path)
    if not raw_results:
        # No reaction detected: return an empty result instead of an IndexError.
        return {}
    raw_pred = raw_results[0]
    return get_reaction_from_raw(raw_pred)




def get_reaction_full(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image, 
    including only reactants, conditions, and products with their smiles, bbox, or text.
    '''
    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    #raw_prediction = get_reaction_withatoms_correctR(image_path)
    return raw_prediction

def get_full_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image = Image.open(image_path).convert('RGB')
    image_file = image_path
    #raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    # Use original data, including complete information like coords and edges
    raw_prediction = get_cached_raw_results(image_path)
    # raw_prediction is a list, each element is a reaction dictionary
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

    #raw_prediction =json.dumps(raw_prediction)
    print(f"raw_prediction:{raw_prediction}")

    # coref_results = model.extract_molecule_corefs_from_figures([image])
    # for item in coref_results:
    #     for bbox in item.get("bboxes", []):
    #         for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
    #             bbox.pop(key, None)  # Safely remove key

    # data = coref_results[0]
    # parsed = parse_coref_data_with_fallback(data)
    
    parsed = get_multi_molecular_text_to_correct(image_path)

    combined_result = {
        "reaction_prediction": raw_prediction,  # is a list
        "molecule_coref": parsed               # structured molecule recognition result
    }
    print(f"combined_result:{combined_result}")
    return combined_result

def get_full_reaction_OS(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image = Image.open(image_path).convert('RGB')
    image_file = image_path
    #raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    # Use original data, including complete information like coords and edges
    raw_prediction = get_cached_raw_results_OS(image_path)
    # raw_prediction is a list, each element is a reaction dictionary
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

    #raw_prediction =json.dumps(raw_prediction)
    print(f"raw_prediction:{raw_prediction}")

    # coref_results = model.extract_molecule_corefs_from_figures([image])
    # for item in coref_results:
    #     for bbox in item.get("bboxes", []):
    #         for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
    #             bbox.pop(key, None)  # Safely remove key

    # data = coref_results[0]
    # parsed = parse_coref_data_with_fallback(data)
    
    parsed = get_multi_molecular_text_to_correct_OS(image_path)

    combined_result = {
        "reaction_prediction": raw_prediction,  # is a list
        "molecule_coref": parsed               # structured molecule recognition result
    }
    print(f"combined_result:{combined_result}")
    return combined_result






def get_full_reaction_template(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image = Image.open(image_path).convert('RGB')
    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    ####################raw_prediction = get_reaction_withatoms_correctR(image_path)###############################################################################################
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

    #raw_prediction =json.dumps(raw_prediction)
    print(f"raw_prediction:{raw_prediction}")
    #coref_results = model.extract_molecule_corefs_from_figures([image])
    coref_results = process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path)
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # Safely remove key

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)

    combined_result = {
        #"reaction_prediction": raw_prediction,  # is a list
        "molecule_coref": parsed               # structured molecule recognition result
    }
    print(f"combined_result:{combined_result}")
    return combined_result

def get_full_reaction_template_OS(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image = Image.open(image_path).convert('RGB')
    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    ####################raw_prediction = get_reaction_withatoms_correctR(image_path)###############################################################################################
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

    #raw_prediction =json.dumps(raw_prediction)
    print(f"raw_prediction:{raw_prediction}")
    #coref_results = model.extract_molecule_corefs_from_figures([image])
    coref_results = process_reaction_image_with_multiple_products_and_text_correctmultiR_OS(image_path)
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # Safely remove key

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)

    combined_result = {
        #"reaction_prediction": raw_prediction,  # is a list
        "molecule_coref": parsed               # structured molecule recognition result
    }
    print(f"combined_result:{combined_result}")
    return combined_result



def process_reaction_image_with_product_variant_R_group(image_path: str) -> dict:
    """
    Input a chemical reaction image path, use GPT and OpenChemIE to extract reaction information, and return organized reaction data.

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

    def encode_image_from_array(img_array: np.ndarray) -> str:
        """
        Convert numpy array (RGB format) to base64 string
        """
        # Ensure uint8 type
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
        
        # Use PIL to convert to PNG byte stream
        img_pil = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        
        # Encode as base64
        return base64.b64encode(img_bytes).decode('utf-8')
    base64_image = encode_image(image_path)

    # GPT tool-calling configuration
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
                {
        'type': 'function',
        'function': {
            'name': 'get_reaction_con',
            'description': 'Get a list of reaction conditions from a reaction image',
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
            }
    ]

    # Message content provided to GPT
    with open('./prompt/prompt_Str_R.txt', 'r', encoding='utf-8') as prompt_file:
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
        'get_multi_molecular_text_to_correct': get_multi_molecular_text_to_correct,
        'get_reaction': get_reaction,
        'get_reaction_con': get_reaction_con
    }

    # Step 2: Handle multiple tool calls
    tool_calls = response.choices[0].message.tool_calls or []

    results = []

    # Iterate through each tool call
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        try:
            tool_args = json.loads(tool_arguments)
        except (json.JSONDecodeError, TypeError):
            # Malformed tool arguments: image_path is used directly below, default to {}.
            tool_args = {}
        
        if tool_name in TOOL_MAP:
            # Call tool and get result
            tool_result = TOOL_MAP[tool_name](image_path)
            print(f"[DEBUG mol_agent] TOOL {tool_name} -> {tool_result}")
        else:
            # Unknown tool name (e.g. VLM drift): skip rather than crash the pipeline.
            print(f"WARNING [mol_agent]: Unknown tool called: {tool_name}, skipping.")
            continue
        
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
    #print(f"tool_results:{tool_result}")

    coref_results = get_cached_multi_molecular(image_path)
    annotated_img = draw_mol_bboxes(image_path, coref_results, output_path=None)
    base64_image_1 = encode_image_from_array(annotated_img)
    
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
                            'url': f'data:image/png;base64,{base64_image_1}'
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
    print("R_group_agent_output:", gpt_output)
    gpt_output = _compensate_missing_molecules(gpt_output, results, 'get_multi_molecular_text_to_correct')
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

 
    #coref_results = model.extract_molecule_corefs_from_figures([image_np])
    #coref_results = process_reaction_image_with_multiple_products_and_text_correctR(image_path)
    #coref_results = get_cached_multi_molecular(image_path)


    # reaction_results = model.extract_reactions_from_figures([image_np])
    #reaction_results = get_reaction_withatoms_correctR(image_path)[0]
    raw_results  = get_cached_raw_results(image_path)
    # No reaction detected: fall back to an empty reaction instead of an IndexError.
    reaction_results = raw_results[0] if raw_results else {}
    
    reaction = {
    "reactants": reaction_results.get('reactants', []),
    "conditions": reaction_results.get('conditions', []),
    "products": reaction_results.get('products', [])
    }
    reaction_results = [{"reactions": [reaction]}]
    #print(reaction_results)
    

    # Define function to update tool output
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

# Get results
    smiles_details = extract_smiles_details(gpt_output, coref_results)
    print(f"[DEBUG mol_agent] smiles_details keys: {list(smiles_details.keys())}")
    print(f"[DEBUG mol_agent] coref_results[0] smiles list: {[b.get('smiles') for it in coref_results for b in it.get('bboxes', [])]}")
    print(f"[DEBUG mol_agent] reaction_results raw: {reaction_results}")

    reactants_array = []
    products = []

    for reactant in reaction_results[0]['reactions'][0]['reactants']:
        if 'smiles' in reactant:
            #print(f"SMILES:{reactant['smiles']}")
            ##print(reactant)
            reactants_array.append(reactant['smiles'])

    for product in reaction_results[0]['reactions'][0]['products']:
        ##print(product['smiles'])
        ##print(product)
        products.append(product['smiles'])
    # Output results
    #import p#print
    #p#print.p#print(smiles_details)

        # Organize reaction data
    backed_out = utils.backout_without_coref(reaction_results, coref_results, gpt_output, smiles_details, model.molnextr)
    print(f"[DEBUG mol_agent] backed_out (before sort) = {backed_out}")
    backed_out.sort(key=lambda x: x[2])
    extracted_rxns = {}
    for reactants, products_, label in backed_out:
        extracted_rxns[label] = {'reactants': reactants, 'products': products_}
    print(f"[DEBUG mol_agent] extracted_rxns = {extracted_rxns}")
    
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # Safely remove key

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)
    
    toadd = {
        "reaction_template": {
            "reactants": reactants_array,
            "products": products
        },
        "reactions": extracted_rxns,
        "original_molecule_list": gpt_output
    }

# Sort by label
    sorted_keys = sorted(toadd["reactions"].keys())
    toadd["reactions"] = {i: toadd["reactions"][i] for i in sorted_keys}
    toadd = normalize_product_variant_output(toadd)
    print(f"str_R_group_agent_output:{toadd}")
    return toadd


def process_reaction_image_with_product_variant_R_group_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = "http://localhost:8000/v1",
    api_key: Optional[str] = None,
) -> dict:
    """
    Aligned with process_reaction_image_with_product_variant_R_group workflow, but uses a local/self-hosted model compatible with OpenAI Chat Completions protocol (such as vLLM or Ollama).

    Args:
        image_path: reaction image path.
        model_name: local model name (default `Qwen/Qwen3-VL-8B-Instruct`).
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
                'name': 'get_multi_molecular_text_to_correct_OS',
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
            'name': 'get_reaction_OS',
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
            {
        'type': 'function',
        'function': {
            'name': 'get_reaction_con_OS',
            'description': 'Get a list of reaction conditions from a reaction image',
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
            }
    ]

    # Message content provided to GPT
    with open('./prompt/prompt_Str_R.txt', 'r', encoding='utf-8') as prompt_file:
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
        max_retries=5,  # Increase retry count because multiple requests may happen simultaneously
        base_delay=3,   # Increase base delay to give the API more recovery time
        backoff_factor=2,
        model=model_name,
        temperature=0,
        #response_format={'type': 'json_object'},
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    # Step 1: Tool mapping table
    TOOL_MAP = {
        'get_multi_molecular_text_to_correct_OS': get_multi_molecular_text_to_correct_OS,
        'get_reaction_OS': get_reaction_OS,
        'get_reaction_con_OS':get_reaction_con_OS
    }

    # Step 2: Handle multiple tool calls
    tool_calls = response.choices[0].message.tool_calls or []
    results = []

    # Iterate through each tool call
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        try:
            tool_args = json.loads(tool_arguments)
        except (json.JSONDecodeError, TypeError):
            # Malformed tool arguments: image_path is used directly below, default to {}.
            tool_args = {}
        
        if tool_name in TOOL_MAP:
            # Call tool and get result
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            # Unknown tool name (e.g. VLM drift): skip rather than crash the pipeline.
            print(f"WARNING [mol_agent]: Unknown tool called: {tool_name}, skipping.")
            continue
        
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
        response_format={'type': 'json_object'},
        temperature=0
    )

    # Get GPT-generated result
    raw_content = response.choices[0].message.content

    # Check whether content is empty
    if not raw_content or not raw_content.strip():
        print(f"ERROR [OS]: Model returned empty content")
        print(f"Full response object: {response}")
        raise ValueError("Model returned empty content. Please check the model response.")

    print(f"DEBUG [OS]: Raw content preview (first 500 chars):\n{raw_content[:500]}")

    # Parse JSON
    gpt_output = None

    try:
        gpt_output = json.loads(raw_content)
        print(f"DEBUG [OS]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        print(f"ERROR [OS]: Failed to parse JSON from model response")
        print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
        raise json.JSONDecodeError(
            f"Could not parse JSON from model response. Content may not be valid JSON.",
            raw_content, 0
        )
    
    print("R_group_agent_output:", gpt_output)
    gpt_output = _compensate_missing_molecules(gpt_output, results, 'get_multi_molecular_text_to_correct')
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Use OS-version caching function
    coref_results = get_cached_multi_molecular_OS(image_path)
    raw_results = get_cached_raw_results_OS(image_path)
    # No reaction detected: fall back to an empty reaction instead of an IndexError.
    reaction_results = raw_results[0] if raw_results else {}
    
    reaction = {
        "reactants": reaction_results.get('reactants', []),
        "conditions": reaction_results.get('conditions', []),
        "products": reaction_results.get('products', [])
    }
    reaction_results = [{"reactions": [reaction]}]

    # Define function to update tool output
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

    # Get results
    smiles_details = extract_smiles_details(gpt_output, coref_results)

    reactants_array = []
    products = []

    for reactant in reaction_results[0]['reactions'][0]['reactants']:
        if 'smiles' in reactant:
            reactants_array.append(reactant['smiles'])

    for product in reaction_results[0]['reactions'][0]['products']:
        products.append(product['smiles'])

    # Organize reaction data
    backed_out = utils.backout_without_coref(reaction_results, coref_results, gpt_output, smiles_details, model.molnextr)
    backed_out.sort(key=lambda x: x[2])
    extracted_rxns = {}
    for reactants, products_, label in backed_out:
        extracted_rxns[label] = {'reactants': reactants, 'products': products_}
    
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]:
                bbox.pop(key, None)  # Safely remove key

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)
    
    toadd = {
        "reaction_template": {
            "reactants": reactants_array,
            "products": products
        },
        "reactions": extracted_rxns,
        "original_molecule_list": gpt_output
    }

    # Sort by label
    sorted_keys = sorted(toadd["reactions"].keys())
    toadd["reactions"] = {i: toadd["reactions"][i] for i in sorted_keys}
    toadd = normalize_product_variant_output(toadd)
    print(f"str_R_group_agent_output:{toadd}")
    return toadd


def process_reaction_image_with_table_R_group(image_path: str) -> dict:

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
    with open('./prompt/prompt_Table_R.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_full_reaction',
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
    tools = tools,
    )

    
    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        # No tool call returned: fall back to the only available tool.
        return get_full_reaction(image_path)
    tool_call = tool_calls[0]
    tool_name = tool_call.function.name  # modify here
    tool_arguments = tool_call.function.arguments  # newly added here
    tool_call_id = tool_call.id

    try:
        tool_args = json.loads(tool_arguments)
    except (json.JSONDecodeError, TypeError):
        tool_args = {}
    #image_path = tool_args.get('image_path', image_path)  # Use image_path provided by model

    if tool_name == 'get_full_reaction':
        tool_result = get_full_reaction(image_path)

    else:
        # Unknown tool name: fall back to the only available tool instead of crashing.
        print(f"WARNING [full_reaction agent]: Unknown tool called: {tool_name}, using get_full_reaction.")
        tool_result = get_full_reaction(image_path)
    #print(tool_result)

    # Build tool-call result message
    function_call_result_message = {
        'role': 'tool',
        'name': tool_name,  # Gemini API requires the name field
        'content': json.dumps({
            'image_path': image_path,
            f'{tool_name}':(tool_result),
    }),
        'tool_call_id': tool_call_id,
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
            response.choices[0].message,
            function_call_result_message,
        ],
    }

    # Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
        #temperature=0
    )

    #print(response)   


    def replace_symbols_and_generate_smiles(input1, input2):
        """
        Generic function to replace symbols from input2 into input1 and generate new SMILES.
        Returned result keeps a specific format and does not include initial reaction data.
        
        Parameters:
        input1: initial input data containing reactants and products
        input2: data containing symbols information for different reactions

        Returns:
        A new dictionary containing each reaction, including reaction_id, reactants, and products.
        """
        
        reactions_output = {"reactions": []}  # store final reaction output
        
        # Iterate over each reaction in input2
        for reaction in input2['reactions']:
            reaction_id = reaction['reaction_id']
            
            # Build new reaction dictionary
            new_reaction = {"reaction_id": reaction_id, "reactants": [], "conditions":[], "products": [], "additional_info": []}

            # Iterate all reactants in input1, keep text type, process molecule type
            mol_idx = 0  # Used to track molecule index in reaction['reactants']
            for j, original_reactant in enumerate(input1['reactants']):
                # If text type, keep directly
                if 'coords' not in original_reactant or 'edges' not in original_reactant:
                    new_reactant = {
                        "category": original_reactant.get('category', '[Txt]'),
                        "bbox": original_reactant.get('bbox', []),
                        "text": original_reactant.get('text', []),
                    }
                    new_reaction["reactants"].append(new_reactant)
                else:
                    # If molecule type, get corresponding symbols from reaction['reactants']
                    if mol_idx < len(reaction['reactants']):
                        reactant = reaction['reactants'][mol_idx]
                        mol_idx += 1
                        
                        new_symbols_reactant = reactant['symbols']  # replace with symbols in reaction
                        new_smiles_reactant, __, __ = _convert_graph_to_smiles(original_reactant['coords'], new_symbols_reactant, original_reactant['edges'])  # generate new SMILES
                        
                        new_reactant = {
                            #"category": original_reactant['category'],
                            #"bbox": original_reactant['bbox'],
                            #"category_id": original_reactant['category_id'],
                            "smiles": new_smiles_reactant,
                            #"coords": original_reactant['coords'],
                            "symbols": new_symbols_reactant,
                            #"edges": original_reactant['edges']
                        }
                        new_reaction["reactants"].append(new_reactant)

            if 'conditions' in reaction:
                new_reaction['conditions'] = reaction['conditions']

            
            # Process each molecule in products
            # Iterate all products in input1, keep text type, process molecule type
            mol_idx = 0  # Used to track molecule index in reaction['products']
            for k, original_product in enumerate(input1['products']):
                # If text type, keep directly
                if 'coords' not in original_product or 'edges' not in original_product:
                    new_product = {
                        "category": original_product.get('category', '[Txt]'),
                        "bbox": original_product.get('bbox', []),
                        "text": original_product.get('text', []),
                    }
                    new_reaction["products"].append(new_product)
                else:
                    # If molecule type, get corresponding symbols from reaction['products']
                    if mol_idx < len(reaction['products']):
                        product = reaction['products'][mol_idx]
                        mol_idx += 1
                        
                        new_symbols_product = product['symbols']  # replace with symbols in reaction
                        new_smiles_product, __, __ = _convert_graph_to_smiles(original_product['coords'], new_symbols_product, original_product['edges'])  # generate new SMILES
                        
                        new_product = {
                            #"category": original_product['category'],
                            #"bbox": original_product['bbox'],
                            #"category_id": original_product['category_id'],
                            "smiles": new_smiles_product,
                            #"coords": original_product['coords'],
                            "symbols": new_symbols_product,
                            #"edges": original_product['edges']
                        }
                        new_reaction["products"].append(new_product)
            
            if 'additional_info' in reaction:
                new_reaction['additional_info'] = reaction['additional_info']

            reactions_output['reactions'].append(new_reaction)  

        return reactions_output
    

    reaction_preds = tool_result['reaction_prediction']
    if isinstance(reaction_preds, str):
        # If it is a string, parse it
        tool_result_json = json.loads(reaction_preds)
    elif isinstance(reaction_preds, (dict, list)):
        # Already dict or list, use directly
        tool_result_json = reaction_preds
    else:
        raise TypeError(f"Unexpected tool_result type: {type(reaction_preds)}")

    input1 = tool_result_json[0]
    input2 = json.loads(response.choices[0].message.content) 
    updated_input = replace_symbols_and_generate_smiles(input1, input2)
    print(f"txt_R_group_agent_output:{updated_input}")
    return updated_input


def process_reaction_image_with_table_R_group_OS(
    image_path: str,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ",
    base_url: Optional[str] = "http://localhost:8000/v1",
    api_key: Optional[str] = None,

) -> dict:
    """
    Aligned with process_reaction_image_with_table_R_group workflow, but uses a local/self-hosted model compatible with OpenAI Chat Completions protocol (such as vLLM or Ollama).

    Args:
        image_path: reaction image path.
        model_name: local model name (default `Qwen/Qwen3-VL-8B-Instruct`).
        base_url: OpenAI-compatible API endpoint; if None, use `http://localhost:8000/v1` (vLLM default port).
        api_key: API key, can be any non-empty string (vLLM default can be `"EMPTY"`).

    Returns:
        dict: organized reaction data including R-group table information.
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
    with open('./prompt/prompt_Table_R.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_full_reaction_OS',
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

    # Call GPT API (with retry mechanism)
    response = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=model_name,
        temperature=0,
        #response_format={'type': 'json_object'},
        messages=[
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
        ],
        tools=tools,
        tool_choice="auto",
    )

    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        # No tool call returned: fall back to the only available tool.
        return get_full_reaction_OS(image_path)
    
    tool_call = tool_calls[0]
    tool_name = tool_call.function.name
    tool_arguments = tool_call.function.arguments
    tool_call_id = tool_call.id

    try:
        tool_args = json.loads(tool_arguments)
    except (json.JSONDecodeError, TypeError):
        tool_args = {}

    if tool_name == 'get_full_reaction_OS':
        tool_result = get_full_reaction_OS(image_path)
    else:
        # Unknown tool name: fall back to the only available tool instead of crashing.
        print(f"WARNING [full_reaction agent OS]: Unknown tool called: {tool_name}, using get_full_reaction_OS.")
        tool_result = get_full_reaction_OS(image_path)

    # Build tool-call result message
    function_call_result_message = {
        'role': 'tool',
        'name': tool_name,  # Gemini API requires the name field
        'content': json.dumps({
            'image_path': image_path,
            f'{tool_name}':(tool_result),
        }),
        'tool_call_id': tool_call_id,
    }

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
            function_call_result_message,
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
        response_format={'type': 'json_object'},
        temperature=0
    )

    print(f"DEBUG [OS]: Model response content type: {type(response.choices[0].message.content)}")
    print(f"DEBUG [OS]: Model response content preview: {str(response.choices[0].message.content)[:500]}")

    def replace_symbols_and_generate_smiles(input1, input2):
        """
        Generic function to replace symbols from input2 into input1 and generate new SMILES.
        Returned result keeps a specific format and does not include initial reaction data.
        
        Parameters:
        input1: initial input data containing reactants and products
        input2: data containing symbols information for different reactions

        Returns:
        A new dictionary containing each reaction, including reaction_id, reactants, and products.
        """
        
        reactions_output = {"reactions": []}  # store final reaction output
        
        # Validate input2 format
        if not isinstance(input2, dict):
            raise ValueError(f"Expected input2 to be a dict, but got {type(input2)}: {input2}")
        
        if 'reactions' not in input2:
            print(f"ERROR [OS]: 'reactions' key not found in input2.")
            print(f"Available keys: {list(input2.keys())}")
            print(f"Full input2 content:\n{json.dumps(input2, indent=2, ensure_ascii=False)}")
            raise KeyError(f"'reactions' key not found in model response. Available keys: {list(input2.keys())}. "
                          f"Please check the model response format and the prompt file './prompt/prompt_Table_R.txt'. "
                          f"The model may not be following the expected JSON schema.")
        
        # Iterate over each reaction in input2
        for reaction in input2['reactions']:
            reaction_id = reaction['reaction_id']
            
            # Build new reaction dictionary
            new_reaction = {"reaction_id": reaction_id, "reactants": [], "conditions":[], "products": [], "additional_info": []}

            # Iterate all reactants in input1, keep text type, process molecule type
            mol_idx = 0  # Used to track molecule index in reaction['reactants']
            for j, original_reactant in enumerate(input1['reactants']):
                # If text type, keep directly
                if 'coords' not in original_reactant or 'edges' not in original_reactant:
                    new_reactant = {
                        "category": original_reactant.get('category', '[Txt]'),
                        "bbox": original_reactant.get('bbox', []),
                        "text": original_reactant.get('text', []),
                    }
                    new_reaction["reactants"].append(new_reactant)
                else:
                    # If molecule type, get corresponding symbols from reaction['reactants']
                    if mol_idx < len(reaction['reactants']):
                        reactant = reaction['reactants'][mol_idx]
                        mol_idx += 1
                        
                        new_symbols_reactant = reactant['symbols']  # replace with symbols in reaction
                        new_smiles_reactant, __, __ = _convert_graph_to_smiles(original_reactant['coords'], new_symbols_reactant, original_reactant['edges'])  # generate new SMILES
                        
                        new_reactant = {
                            "smiles": new_smiles_reactant,
                            "symbols": new_symbols_reactant,
                        }
                        new_reaction["reactants"].append(new_reactant)

            if 'conditions' in reaction:
                new_reaction['conditions'] = reaction['conditions']

            
            # Process each molecule in products
            # Iterate all products in input1, keep text type, process molecule type
            mol_idx = 0  # Used to track molecule index in reaction['products']
            for k, original_product in enumerate(input1['products']):
                # If text type, keep directly
                if 'coords' not in original_product or 'edges' not in original_product:
                    new_product = {
                        "category": original_product.get('category', '[Txt]'),
                        "bbox": original_product.get('bbox', []),
                        "text": original_product.get('text', []),
                    }
                    new_reaction["products"].append(new_product)
                else:
                    # If molecule type, get corresponding symbols from reaction['products']
                    if mol_idx < len(reaction['products']):
                        product = reaction['products'][mol_idx]
                        mol_idx += 1
                        
                        new_symbols_product = product['symbols']  # replace with symbols in reaction
                        new_smiles_product, __, __ = _convert_graph_to_smiles(original_product['coords'], new_symbols_product, original_product['edges'])  # generate new SMILES
                        
                        new_product = {
                            "smiles": new_smiles_product,
                            "symbols": new_symbols_product,
                        }
                        new_reaction["products"].append(new_product)
            
            if 'additional_info' in reaction:
                new_reaction['additional_info'] = reaction['additional_info']

            reactions_output['reactions'].append(new_reaction)  

        return reactions_output
    

    reaction_preds = tool_result['reaction_prediction']
    if isinstance(reaction_preds, str):
        # If it is a string, parse it
        tool_result_json = json.loads(reaction_preds)
    elif isinstance(reaction_preds, (dict, list)):
        # Already dict or list, use directly
        tool_result_json = reaction_preds
    else:
        raise TypeError(f"Unexpected tool_result type: {type(reaction_preds)}")

    input1 = tool_result_json[0]
    
    # Get raw content returned by model
    raw_content = response.choices[0].message.content
    
    # Check whether content is empty
    if not raw_content or not raw_content.strip():
        print(f"ERROR [OS]: Model returned empty content")
        print(f"Full response object: {response}")
        raise ValueError("Model returned empty content. Please check the model response.")
    
    print(f"DEBUG [OS]: Raw content type: {type(raw_content)}")
    print(f"DEBUG [OS]: Raw content length: {len(raw_content)}")
    print(f"DEBUG [OS]: Raw content preview (first 500 chars):\n{raw_content[:500]}")
    
    # Parse JSON
    input2 = None

    try:
        input2 = json.loads(raw_content)
        print(f"DEBUG [OS]: Successfully parsed JSON directly")
    except json.JSONDecodeError:
        print(f"ERROR [OS]: Failed to parse JSON from model response")
        print(f"Raw content (last 2000 chars):\n{raw_content[-2000:]}")
        raise json.JSONDecodeError(
            f"Could not parse JSON from model response. Content may not be valid JSON.",
            raw_content, 0
        )

    # Validate format of input2
    print(f"DEBUG [OS]: input2 type: {type(input2)}")
    if isinstance(input2, dict):
        print(f"DEBUG [OS]: input2 keys: {list(input2.keys())}")
        print(f"DEBUG [OS]: input2 content preview (first 1000 chars):\n{json.dumps(input2, indent=2, ensure_ascii=False)[:1000]}")
    else:
        print(f"DEBUG [OS]: input2 is not a dict, value: {input2}")
    
    updated_input = replace_symbols_and_generate_smiles(input1, input2)
    print(f"txt_R_group_agent_output:{updated_input}")
    return updated_input
