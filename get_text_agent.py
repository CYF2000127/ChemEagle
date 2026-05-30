from PIL import Image
import pytesseract
from chemrxnextractor import RxnExtractor
from openai import AzureOpenAI, OpenAI
from typing import Optional
model_dir = "./cre_models_v0.1"
rxn_extractor = RxnExtractor(model_dir)
import json
import torch
from chemiener import ChemNER
from huggingface_hub import hf_hub_download
ckpt_path = "./ner.ckpt"
model2 = ChemNER(ckpt_path, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
import base64
import os
import shutil
import re
import time
from openai import InternalServerError, RateLimitError, APIError


API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")


# Configure Tesseract OCR path (Windows)
def configure_tesseract():
    """Automatically detect and configure the Tesseract OCR executable path"""
    # If already configured, return directly
    if hasattr(pytesseract.pytesseract, 'tesseract_cmd') and pytesseract.pytesseract.tesseract_cmd:
        if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            return
    
    # Common Windows installation paths (including custom paths under the project directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        # User-specified absolute path (highest priority)
        r"F:\chemeagle\Tesseract-OCR\tesseract.exe",
        # Custom path under the project directory
        os.path.join(script_dir, "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.path.dirname(script_dir), "Tesseract-OCR", "tesseract.exe"),
        # Standard installation path
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        r"C:\Users\Administrator\AppData\Local\Tesseract-OCR\tesseract.exe",
    ]
    
    # First try to find it in PATH
    try:
        tesseract_cmd = shutil.which("tesseract")
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            print(f"✓ Found Tesseract in PATH: {tesseract_cmd}")
            return
    except Exception:
        pass
    
    # If not found in PATH, try common paths
    for path in possible_paths:
        # Normalize path
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            pytesseract.pytesseract.tesseract_cmd = normalized_path
            print(f"✓ Found Tesseract: {normalized_path}")
            return
    
    # If still not found, prompt the user
    print("⚠️  Warning: Tesseract OCR executable not found")
    print("Paths tried:")
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        exists = "✓" if os.path.exists(normalized_path) else "✗"
        print(f"  {exists} {normalized_path}")
    print("\nPlease do one of the following:")
    print("1. Make sure Tesseract OCR is installed correctly")
    print("2. Or set the path manually:")
    print("   pytesseract.pytesseract.tesseract_cmd = r'F:\\chemeagle\\Tesseract-OCR\\tesseract.exe'")
    raise FileNotFoundError(
        "Tesseract OCR is not installed or not in PATH."
        "Please visit https://github.com/UB-Mannheim/tesseract/wiki for installation."
    )

# Initialize Tesseract configuration
configure_tesseract()


def merge_sentences(sentences):
    """
    Merge a list of sentence fragments into a coherent paragraph string.
    """
    # Trim whitespace around each fragment and remove empty strings
    cleaned = [s.strip() for s in sentences if s.strip()]
    # Join with spaces to reconstruct a full paragraph
    paragraph = [" ".join(cleaned)]
    return paragraph


def split_text_into_sentences(text: str) -> list:
    """
    Split text into sentences to avoid issues caused by overly long text.
    Use simple punctuation-based splitting while preserving sentence boundaries.
    """
    # Split by periods, question marks, and exclamation marks, while keeping punctuation
    sentences = re.split(r'([.!?]+)', text)
    # Merge punctuation with preceding text
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
        else:
            sentence = sentences[i].strip()
        if sentence:
            result.append(sentence)
    
    # If no sentence boundary is found, try splitting by newlines
    if not result:
        result = [line.strip() for line in text.splitlines() if line.strip()]
    
    # If still not found, return the whole text (with length limit)
    if not result:
        # Limit single sentence length to avoid exceeding model limits
        max_length = 500  # character limit
        if len(text) > max_length:
            # Split by spaces into smaller chunks
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > max_length and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            result = chunks
        else:
            result = [text]
    
    return result


def extract_reactions_from_text_in_image(image_path: str) -> dict:
    """
    Extract text from a chemical reaction image and identify reactions.

    Arguments:
      image_path: image file path

    Returns:
      {
        'raw_text': full text extracted by OCR (str),
        'paragraph': merged paragraph text (str),
        'reactions': reaction list output by RxnExtractor (list)
      }
    """
    # Model directory and device parameters (adjust as needed)
    model_dir = "./cre_models_v0.1"
    device = "cpu"

    # 1. OCR text extraction
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    # 2. Merge multi-line text into a single paragraph
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    # 3. Split text into sentences to avoid length issues
    sentences = split_text_into_sentences(paragraph)
    
    # 4. Initialize chemical reaction extractor
    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)

    # 5. Extract reactions for each sentence (avoid length mismatch issues)
    all_reactions = []
    try:
        reactions = rxn_extractor.get_reactions(sentences)
        all_reactions = reactions
    except AssertionError as e:
        # If it still fails, try processing sentence by sentence
        print(f"Warning: batch processing failed, trying sentence-by-sentence processing: {e}")
        all_reactions = []
        for sent in sentences:
            try:
                sent_reactions = rxn_extractor.get_reactions([sent])
                all_reactions.extend(sent_reactions)
            except Exception as sent_e:
                print(f"Warning: skipping sentence (processing failed): {sent[:50]}... Error: {sent_e}")
                continue

    return all_reactions 

def NER_from_text_in_image(image_path: str) -> dict:
    # Model directory and device parameters (adjust as needed)
    model_dir = "./cre_models_v0.1"
    device = "cpu"

    # 1. OCR text extraction
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    # 2. Merge multi-line text into a single paragraph
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    # 3. Initialize chemical reaction extractor
    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)

    # 4. Extract reactions (note: get_reactions requires list input)
    predictions = model2.predict_strings([paragraph])

    return predictions 




def text_extraction_agent(image_path: str, graphical_input: Optional[dict] = None) -> dict:
    """
    Agent that calls two tools:
      1) extract_reactions_from_text_in_image
      2) NER_from_text_in_image
    to perform OCR, reaction extraction, and chemical NER on a single image.
    Returns a merged JSON result.
    """
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    # Encode image as Base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Define tools for the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_reactions_from_text_in_image",
                "description": "OCR image and extract chemical reactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "NER_from_text_in_image",
                "description": "OCR image and perform chemical named entity recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        }
    ]

    # Prompt instructing to call both tools
    prompt = """
In this input reaction graphic, there is a chemical reaction scheme template with multiple product/fragment molecular sub-images or tables, conditions, and a text description underneath (or you will receive direct text input).
Your task is to:

1. **Entity Extraction**: First call the `MolNER`(“NER_from_text_in_image") tool to extract chemical identities (names, formulas, abbreviations) from the text description.
2. **Reaction Extraction**: Call the `RxnConInterpreter`("extract_reactions_from_text_in_image") tool to identify reaction types and relationships in the text description.
3. **Correction & Alignment**: 
   - Check the tool results and correct any errors.
   - Align the identified text entities with the graphical output (e.g., mapping "precatalyst B27" in text to the molecule B27 in the image).
4. **Output**: Your final output must be in JSON format. Use specific tags like `[reactant]`, `[reagent]`, `[ABBREVIATION]`, `[SMILES:...]` to annotate the original text.

### Example 1
**User Input:**
Text Input: "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates and oxaziridines was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes were used as precursors of azolium enolates. A bifunctional NHC precatalyst B27 bearing a free hydroxyl group was employed."

**Assistant Response:**
Here is my step-by-step analysis:

**Step 1: Entity Extraction (MolNER)**
* **Observation:**
    * "NHC-bound azolium enolates" -> Chemical (Plural/Class)
    * "oxaziridines" -> Chemical (Systematic Name)
    * "Aryl(alkyl)-disubstituted ketenes" -> Chemical (Reactant precursor)
    * "NHC" -> Abbreviation
    * "B27" -> Chemical (Identifier/Reagent)
* **Action:** Tag these entities with their specific roles and types.

**Step 2: Reaction & Structure Interpretation (RxnConInterpreter)**
* **Observation:**
    * "B27" is identified as a specific precatalyst.
    * **Action:** Retrieve/Associate the SMILES for B27 from the tool/image context: `CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F`.
    * "azolium enolates" and "oxaziridines" are participating reactants in a [3+2] cycloaddition.

**Step 3: Text Alignment & Annotation**
* **Thought:** I need to inject the extracted tags and SMILES directly into the original text string to create the final description.
* **Action:** Construct the annotated string.

**Final JSON Output:**
```json
{
  "annotated_text": "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates [reactant][MULTIPLE] and oxaziridines [reactant][SYSTEMATIC] was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes [reactant] were used as precursors of azolium enolates. A bifunctional NHC [ABBREVIATION] precatalyst B27 [reagent][IDENTIFIERS][SMILES:CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F] bearing a free hydroxyl group was employed."
}
"""

    if graphical_input:
        prompt += f"\n\nGraphical extraction results from the reaction image (use this for alignment with text entities):\n{json.dumps(graphical_input, ensure_ascii=False, indent=2)}"

    messages = [
        {"role": "system", "content": "You are the Text Extraction Agent. Your task is to extract text descriptions from chemical reaction images (or process direct text input), identify chemical entities and reactions within that text, and output a structured annotation."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    # First API call: let GPT decide which tools to invoke
    response1 = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        tools=tools,
        #temperature=0,
        response_format={"type": "json_object"}
    )

    # Get assistant message with tool calls
    assistant_message = response1.choices[0].message
    
    # Execute each requested tool
    tool_calls = assistant_message.tool_calls
    if not tool_calls:
        # If no tool calls, return the response directly
        return json.loads(response1.choices[0].message.content) if response1.choices[0].message.content else {}
    
    tool_results_msgs = []
    for call in tool_calls:
        name = call.function.name
        tool_call_id = call.id
        
        if name == "extract_reactions_from_text_in_image":
            result = extract_reactions_from_text_in_image(image_path)
        elif name == "NER_from_text_in_image":
            result = NER_from_text_in_image(image_path)
        else:
            continue
        
        # Correct format for tool messages: need tool_call_id, not tool_name
        tool_results_msgs.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result, ensure_ascii=False)
        })

    # Second API call: pass tool outputs back to GPT for final response
    # Add assistant message and tool results to messages
    messages.append(assistant_message)
    messages.extend(tool_results_msgs)
    
    response2 = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        #   temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response2.choices[0].message.content)


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


def text_extraction_agent_OS(
    image_path: str,
    *,
    graphical_input: Optional[dict] = None,
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

    # Encode image as Base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Define tools for the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_reactions_from_text_in_image",
                "description": "OCR image and extract chemical reactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "NER_from_text_in_image",
                "description": "OCR image and perform chemical named entity recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                }
            }
        }
    ]

    # Prompt instructing to call both tools
    prompt = """
In this input reaction graphic, there is a chemical reaction scheme template with multiple product/fragment molecular sub-images or tables, conditions, and a text description underneath (or you will receive direct text input).
Your task is to:

1. **Entity Extraction**: First call the `MolNER`("NER_from_text_in_image") tool to extract chemical identities (names, formulas, abbreviations) from the text description.
2. **Reaction Extraction**: Call the `RxnConInterpreter`("extract_reactions_from_text_in_image") tool to identify reaction types and relationships in the text description.
3. **Correction & Alignment**: 
   - Check the tool results and correct any errors.
   - Align the identified text entities with the graphical output (e.g., mapping "precatalyst B27" in text to the molecule B27 in the image).
4. **Output**: Your final output must be in JSON format. Use specific tags like `[reactant]`, `[reagent]`, `[ABBREVIATION]`, `[SMILES:...]` to annotate the original text.

### Example 1
**User Input:**
Text Input: "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates and oxaziridines was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes were used as precursors of azolium enolates. A bifunctional NHC precatalyst B27 bearing a free hydroxyl group was employed."

**Assistant Response:**
Here is my step-by-step analysis:

**Step 1: Entity Extraction (MolNER)**
* **Observation:**
    * "NHC-bound azolium enolates" -> Chemical (Plural/Class)
    * "oxaziridines" -> Chemical (Systematic Name)
    * "Aryl(alkyl)-disubstituted ketenes" -> Chemical (Reactant precursor)
    * "NHC" -> Abbreviation
    * "B27" -> Chemical (Identifier/Reagent)
* **Action:** Tag these entities with their specific roles and types.

**Step 2: Reaction & Structure Interpretation (RxnConInterpreter)**
* **Observation:**
    * "B27" is identified as a specific precatalyst.
    * **Action:** Retrieve/Associate the SMILES for B27 from the tool/image context: `CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F`.
    * "azolium enolates" and "oxaziridines" are participating reactants in a [3+2] cycloaddition.

**Step 3: Text Alignment & Annotation**
* **Thought:** I need to inject the extracted tags and SMILES directly into the original text string to create the final description.
* **Action:** Construct the annotated string.

**Final JSON Output:**
```json
{
  "annotated_text": "In 2010, an enantioselective formal [3+2] cycloaddition of NHC-bound azolium enolates [reactant][MULTIPLE] and oxaziridines [reactant][SYSTEMATIC] was described by Ye and co-workers. Aryl(alkyl)-disubstituted ketenes [reactant] were used as precursors of azolium enolates. A bifunctional NHC [ABBREVIATION] precatalyst B27 [reagent][IDENTIFIERS][SMILES:CCCC(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F] bearing a free hydroxyl group was employed."
}
```

"""

    if graphical_input:
        prompt += f"\n\nGraphical extraction results from the reaction image (use this for alignment with text entities):\n{json.dumps(graphical_input, ensure_ascii=False, indent=2)}"

    messages = [
        {"role": "system", "content": "You are the Text Extraction Agent. Your task is to extract text descriptions from chemical reaction images (or process direct text input), identify chemical entities and reactions within that text, and output a structured annotation."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    # First API call: let GPT decide which tools to invoke
    # Note: vLLM may not support response_format and tools simultaneously
    try:
        response1 = retry_api_call(
            client.chat.completions.create,
            max_retries=5,
            base_delay=3,
            backoff_factor=2,
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            # response_format={"type": "json_object"},  # vLLM does not support using response_format and tools simultaneously
        )
    except Exception as e:
        error_msg = str(e)
        if "tool" in error_msg.lower() or "tool-call" in error_msg.lower():
            print(f"⚠️ Warning: vLLM does not support tool calling: {e}")
            print("Tip: restart the vLLM container with the following arguments:")
            print("  --enable-auto-tool-choice --tool-call-parser auto")
            print("Or continue using Ollama (native tool-calling support)")
            raise
        else:
            raise

    # Get assistant message with tool calls
    assistant_message = response1.choices[0].message
    
    # Execute each requested tool
    tool_calls = assistant_message.tool_calls
    if not tool_calls:
        # If no tool calls, try to parse response directly
        raw_content = response1.choices[0].message.content
        if raw_content:
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                try:
                    from get_R_group_sub_agent import extract_json_from_text_with_reasoning
                    result = extract_json_from_text_with_reasoning(raw_content)
                    if result is not None:
                        return result
                except ImportError:
                    pass
                return {"content": raw_content}
        return {}
    
    tool_results_msgs = []
    for call in tool_calls:
        name = call.function.name
        tool_call_id = call.id
        
        if name == "extract_reactions_from_text_in_image":
            result = extract_reactions_from_text_in_image(image_path)
        elif name == "NER_from_text_in_image":
            result = NER_from_text_in_image(image_path)
        else:
            continue
        
        # Correct format for tool messages: need tool_call_id and name (for some APIs)
        tool_results_msgs.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,  # Some APIs (like Gemini) require name field
            "content": json.dumps(result, ensure_ascii=False)
        })

    # Second API call: pass tool outputs back to GPT for final response
    # Add assistant message and tool results to messages
    messages.append(assistant_message)
    messages.extend(tool_results_msgs)
    
    response2 = retry_api_call(
        client.chat.completions.create,
        max_retries=5,
        base_delay=3,
        backoff_factor=2,
        model=model_name,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw_content = response2.choices[0].message.content
    
    return raw_content
