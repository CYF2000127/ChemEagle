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
model2 = ChemNER(ckpt_path, device=torch.device('cpu'))
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


# 配置 Tesseract OCR 路径（Windows）
def configure_tesseract():
    """自动检测并配置 Tesseract OCR 可执行文件路径"""
    # 如果已经配置过，直接返回
    if hasattr(pytesseract.pytesseract, 'tesseract_cmd') and pytesseract.pytesseract.tesseract_cmd:
        if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            return
    
    # 常见的 Windows 安装路径（包括项目目录下的自定义路径）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        # 用户指定的绝对路径（最高优先级）
        r"F:\chemeagle\Tesseract-OCR\tesseract.exe",
        # 项目目录下的自定义路径
        os.path.join(script_dir, "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.path.dirname(script_dir), "Tesseract-OCR", "tesseract.exe"),
        # 标准安装路径
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        r"C:\Users\Administrator\AppData\Local\Tesseract-OCR\tesseract.exe",
    ]
    
    # 首先尝试从 PATH 中查找
    try:
        tesseract_cmd = shutil.which("tesseract")
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            print(f"✓ 从 PATH 中找到 Tesseract: {tesseract_cmd}")
            return
    except Exception:
        pass
    
    # 如果 PATH 中没有，尝试常见路径
    for path in possible_paths:
        # 规范化路径
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            pytesseract.pytesseract.tesseract_cmd = normalized_path
            print(f"✓ 找到 Tesseract: {normalized_path}")
            return
    
    # 如果都没找到，提示用户
    print("⚠️  警告: 未找到 Tesseract OCR 可执行文件")
    print("已尝试的路径:")
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        exists = "✓" if os.path.exists(normalized_path) else "✗"
        print(f"  {exists} {normalized_path}")
    print("\n请执行以下步骤之一:")
    print("1. 确保 Tesseract OCR 已正确安装")
    print("2. 或者手动设置路径:")
    print("   pytesseract.pytesseract.tesseract_cmd = r'F:\\chemeagle\\Tesseract-OCR\\tesseract.exe'")
    raise FileNotFoundError(
        "Tesseract OCR 未安装或不在 PATH 中。"
        "请访问 https://github.com/UB-Mannheim/tesseract/wiki 下载安装。"
    )

# 初始化 Tesseract 配置
configure_tesseract()


def merge_sentences(sentences):
    """
    合并一个句子片段列表为一个连贯的段落字符串。
    """
    # 去除每条片段前后空白，并剔除空串
    cleaned = [s.strip() for s in sentences if s.strip()]
    # 用空格拼接，恢复成完整段落
    paragraph = [" ".join(cleaned)]
    return paragraph


def split_text_into_sentences(text: str) -> list:
    """
    将文本分割成句子，避免文本过长导致的问题。
    使用简单的标点符号分割，保留句子边界。
    """
    # 按句号、问号、感叹号分割，但保留这些标点
    sentences = re.split(r'([.!?]+)', text)
    # 合并标点和前面的文本
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
        else:
            sentence = sentences[i].strip()
        if sentence:
            result.append(sentence)
    
    # 如果没有找到句子边界，尝试按换行符分割
    if not result:
        result = [line.strip() for line in text.splitlines() if line.strip()]
    
    # 如果还是没有，返回整个文本（但限制长度）
    if not result:
        # 限制单个句子长度，避免超过模型限制
        max_length = 500  # 字符数限制
        if len(text) > max_length:
            # 按空格分割成更小的块
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
    从化学反应图像中提取文本并识别反应。

    参数：
      image_path: 图像文件路径

    返回：
      {
        'raw_text': OCR 提取的完整文本（str),
        'paragraph': 合并后的段落文本 (str),
        'reactions': RxnExtractor 输出的反应列表 (list)
      }
    """
    # 模型目录和设备参数（可按需修改）
    model_dir = "./cre_models_v0.1"
    device = "cpu"

    # 1. OCR 提取文本
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    # 2. 将多行文本合并为单段落
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    # 3. 将文本分割成句子，避免长度问题
    sentences = split_text_into_sentences(paragraph)
    
    # 4. 初始化化学反应提取器
    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)

    # 5. 对每个句子提取反应（避免长度不匹配问题）
    all_reactions = []
    try:
        reactions = rxn_extractor.get_reactions(sentences)
        all_reactions = reactions
    except AssertionError as e:
        # 如果还是出错，尝试逐个句子处理
        print(f"警告: 批量处理失败，尝试逐个句子处理: {e}")
        all_reactions = []
        for sent in sentences:
            try:
                sent_reactions = rxn_extractor.get_reactions([sent])
                all_reactions.extend(sent_reactions)
            except Exception as sent_e:
                print(f"警告: 跳过句子（处理失败）: {sent[:50]}... 错误: {sent_e}")
                continue

    return all_reactions 

def NER_from_text_in_image(image_path: str) -> dict:
    # 模型目录和设备参数（可按需修改）
    model_dir = "./cre_models_v0.1"
    device = "cpu"

    # 1. OCR 提取文本
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    # 2. 将多行文本合并为单段落
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    # 3. 初始化化学反应提取器
    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)

    # 4. 提取反应（注意 get_reactions 需要列表输入）
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
    通用的 API 调用重试函数，支持指数退避策略。
    
    Args:
        func: 要调用的函数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        backoff_factor: 退避因子（每次重试延迟时间 = base_delay * backoff_factor^attempt）
        *args, **kwargs: 传递给 func 的参数
    
    Returns:
        func 的返回值
    
    Raises:
        最后一次尝试的异常
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (InternalServerError, RateLimitError, APIError) as e:
            last_exception = e
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            error_message = str(e)
            
            # 检查是否是 503 错误或其他可重试的错误
            if error_code == 503 or 'overloaded' in error_message.lower() or '503' in error_message:
                if attempt < max_retries - 1:
                    delay = base_delay * (backoff_factor ** attempt)
                    print(f"⚠️ API 调用失败 (503/过载)，第 {attempt + 1}/{max_retries} 次尝试。{delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ API 调用失败，已达到最大重试次数 ({max_retries})")
                    raise
            else:
                # 其他类型的错误，直接抛出
                raise
        except Exception as e:
            # 其他未知错误，直接抛出
            raise
    
    # 如果所有重试都失败了
    if last_exception:
        raise last_exception
    raise RuntimeError("API 调用失败，未知错误")


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
            # response_format={"type": "json_object"},  # vLLM 不支持同时使用 response_format 和 tools
        )
    except Exception as e:
        error_msg = str(e)
        if "tool" in error_msg.lower() or "tool-call" in error_msg.lower():
            print(f"⚠️ 警告: vLLM 不支持工具调用: {e}")
            print("提示: 请重新启动 vLLM 容器，添加以下参数:")
            print("  --enable-auto-tool-choice --tool-call-parser auto")
            print("或者继续使用 Ollama（原生支持工具调用）")
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
        # response_format={"type": "json_object"},  # vLLM 可能不支持
    )

    # Parse response (support extracting JSON from text with reasoning)
    raw_content = response2.choices[0].message.content
    
    try:
        # First try direct JSON parsing
        return json.loads(raw_content)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from text
        try:
            from get_R_group_sub_agent import extract_json_from_text_with_reasoning
            result = extract_json_from_text_with_reasoning(raw_content)
            if result is not None:
                return result
        except ImportError:
            pass
        
        # If all else fails, return raw content wrapped in dict
        print(f"⚠️ 警告: 无法解析 JSON，返回原始内容")
        print(f"Raw content (last 500 chars):\n{raw_content[-500:]}")
        return {"content": raw_content}

