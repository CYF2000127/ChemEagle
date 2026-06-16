<div align="center">
  <img src="examples/chemeagle_logo2.png" width="250" alt="ChemEAGLE Logo">
  <h1>ChemEAGLE</h1>
</div>




![visualization](examples/chemagle_overview_v10.png)
<div align="center",width="100">
</div> 

This is the official code of the [paper](https://arxiv.org/abs/2507.20230) "A Multi-Agent System Enables Versatile Information Extraction from the Chemical Literature".



## :sparkles: Highlights
<p align="justify">
In this work, we present ChemEAGLE, a multimodal large language model (MLLM)-based multi-agent system that integrates diverse chemical information extraction tools to extract multimodal chemical reactions. By integrating ten expert-designed tools and seven chemical information extraction agents, ChemEAGLE not only processes individual modalities but also utilizes MLLMs' reasoning capabilities to unify extracted data, ensuring more accurate and comprehensive reaction representations. By bridging multimodal gaps, our approach significantly improves automated chemical knowledge extraction, facilitating more robust AI-driven chemical research.

[comment]: <> ()
![visualization](examples/fig_2.png)
<div align="center"> An example workflow of ChemEAGLE. Each agent handles a specific sub-task, from reaction template parsing and molecular recognition to SMILES reconstruction and condition role interpretation, ensuring accurate, structured chemical data integration. </div>
  
### 🧩 Agents Overview
| Agent Name                                          | Category            | Main Function                                                       |
| --------------------------------------------------- | ------------------- | ------------------------------------------------------------------- |
| **Planner**                                   | Planning  | Analyzes input, plans extraction steps, assigns sub-tasks to agents |
| **Plan Observer**                          | Validation | Monitors extraction workflow, ensures logical plan                  |
| **Action Observer**                           | Validation | Oversees agent actions, validates consistency and correctness       |
| **Reaction Template Parsing Agent**                 | Extraction          | Parses reaction templates, integrates R-group substitutions         |
| **Molecular Recognition Agent**                     | Extraction          | Detects and interprets all molecules in graphics                      |
| **Structure-based Table R-group Substitution Agent** | Extraction          | Substitutes R-groups and reconstructs reactant SMILES from product variant structure-based tables        |
| **Text-based Table R-group Substitution Agent**     | Extraction          | Substitutes R-groups and reconstructs SMILES from text-based tables  |
| **Condition Interpretation Agent**                  | Extraction          | Extracts and categorizes reaction conditions (solvent, temp, etc.)  |
| **Text Extraction Agent**                           | Extraction          | Extracts and aligns reaction info from associated texts             |
| **Data Structure Agent**                            | Output         | Compiles structured output for downstream applications              |


### 🛠️ Toolkits and Web Services Used in ChemEAGLE
| Tool Name               | Category                          | Description                                            |
| ----------------------- | --------------------------------- | ------------------------------------------------------ |
| **TesseractOCR**        | Computer Vision                   | Optical character recognition for text in graphics       |
| **TableParser**         | Computer Vision                   | Table structure detection and parsing                  |
| **MolDetector**         | Computer Vision                   | Locates and segments molecules within graphics           |
| **Image2Graph**         | Molecular Recognition             | Converts molecular sub-images to graph representations     |
| **Graph2SMILES**        | Molecular Recognition             | Converts molecular graphs to SMILES strings            |
| **SMILESReconstructor** | Molecular Recognition             | Reconstructs reactant SMILES from product variants     |
| **RxnImgParser**        | Reaction Image Parsing            | Parsing reaction template images into bounding boxes and components |
| **RxnConInterpreter**   | Reaction Image Parsing            | Assigns condition roles to extracted condition text     |
| **MolNER**              | Text-based Information Extraction | Chemical named entity recognition from text            |
| **ChemRxnExtractor**    | Text-based Information Extraction | Extracts chemical reactions and roles from text        |
|**OPSIN** |  Web Service for Name2SMILES                     | Converts chemical names to SMILES strings https://opsin.ch.cam.ac.uk/opsin/ |
|**PUBCHEM**|      Web Service for Name2SMILES                | Converts chemical names to SMILES strings https://pubchem.ncbi.nlm.nih.gov/rest/pug|
|**CIR** |       Web Service for Name2SMILES                  | Converts chemical names to SMILES strings https://cactus.nci.nih.gov/chemical/structure|
| **FormulaSolver**       | Formula2Name                       | Converts chemical formula into chemical names that can be queried by Web Service |



## :rocket: Using the code for ChemEAGLE
### Using the code
Clone the following repositories:
```
git clone https://github.com/CYF2000127/ChemEagle
```
#### Option A: Using Azure OpenAI (Cloud-based)

1. First create and activate a [conda](https://numdifftools.readthedocs.io/en/stable/how-to/create_virtual_env_with_conda.html) environment with the following command in a Linux, Windows, or MacOS environment (Linux is the most recommended):
```bash
conda create -n chemeagle python=3.10
conda activate chemeagle
```

2. Then install requirements:
```bash
pip install -r requirements.txt
```

3. Download the necessary [models](https://huggingface.co/CYF200127/ChemEAGLEModel/tree/main) and put in the main path.

4. Set up your Azure OpenAI API key in your environment. Here are two detailed tutorials ([Chinese Version](https://zhuanlan.zhihu.com/p/678367436), [English Version](https://www.datacamp.com/tutorial/azure-openai)) on how to obtain the Azure OpenAI API key and endpoint (Remember to use the API key and the endpoint in the Azure AI Studio).
```bash
export API_KEY=your-azure-openai-api-key
export AZURE_ENDPOINT=your-azure-endpoint
export API_VERSION=your-api-version
```

5. Run the following code to extract machine-readable chemical data from chemical graphics:
```python
from main import ChemEagle
image_path = './examples/1.png'
results = ChemEagle(image_path)
print(results)
```
All implementations also can run on the colab, we provided a example code [here](https://colab.research.google.com/drive/1pOrBPm_QYgZgeKIDbGULGjsyTuOx5nfD#scrollTo=s-7RdEIbAkvr).

6. Alternatively, run the following code to extract machine-readable chemical data from chemical literature (PDF files) directly:
```python
import os
from main import ChemEagle
from pdf_extraction import run_pdf
pdf_path   = 'your/pdf/path'
output_dir = 'your/output/dir'
run_pdf(pdf_dir=pdf_path, image_dir=output_dir)
results = []
for fname in sorted(os.listdir(output_dir)):
    if not fname.lower().endswith('.png'):
        continue
    img_path = os.path.join(output_dir, fname)
    try:
        r = ChemEagle(img_path)
        r['image_name'] = fname
        results.append(r)
    except Exception as e:
        results.append({'image_name': fname, 'error': str(e)})
print(results)
```

#### Option B: Using ChemEagle_OS (Local Deployment with vLLM)

**ChemEagle_OS** is an open-source version that runs locally using vLLM, eliminating the need for cloud API keys.

##### Prerequisites
- NVIDIA GPU with CUDA support (recommended)
- Docker installed (for Windows vLLM deployment)
- Download the Qwen3-VL or Qwen3.5 series model weights (We provided `Qwen3-VL-32B-Instruct` and `Qwen3-VL-32B-Instruct-AWQ`) from [HuggingFace](https://huggingface.co/CYF200127/Qwen3-VL-32B-Instruct).

##### Hardware Requirements (VRAM)
Depending on the model size and architecture (Dense vs. MoE), the VRAM requirements vary significantly. Below are the estimated minimum physical VRAM requirements for the Qwen3-VL and Qwen3.5 series, including both BF16 and FP8/INT8 quantized versions. 

###### Qwen3-VL Series

| Model Version | Architecture | BF16 VRAM | FP8/INT8 VRAM |
| :--- | :--- | :--- | :--- |
| **Qwen3-VL-2B** | Dense | ~4-5 GB | **~2-3 GB** |
| **Qwen3-VL-4B** | Dense | ~8-10 GB | **~4-5 GB** |
| **Qwen3-VL-8B** | Dense | ~16-20 GB | **~8-10 GB** |
| **Qwen3-VL-30B-A3B**| MoE | ~60-75 GB | **~30-38 GB** |
| **Qwen3-VL-32B** | Dense | ~64-80 GB | **~32-40 GB** |
| **Qwen3-VL-235B-A22B**| MoE | ~450-550 GB | **~225-275 GB** |

###### Qwen3.5 Series

| Model Version | Architecture | BF16 VRAM | FP8/INT8 VRAM |
| :--- | :--- | :--- | :--- |
| **Qwen3.5-0.8B / 2B** | Dense | ~3-5 GB | **< 3 GB** |
| **Qwen3.5-4B** | Dense | ~8-10 GB | **~4.5-6 GB** |
| **Qwen3.5-9B** | Dense | ~18-20 GB | **~9-11 GB** |
| **Qwen3.5-27B** | Dense | ~54-60 GB | **~27-32 GB** |
| **Qwen3.5-35B-A3B** | MoE | ~70-75 GB | **~35-40 GB** |
| **Qwen3.5-122B-A10B** | MoE | ~245-260 GB | **~122-135 GB** |
| **Qwen3.5-397B-A17B** | MoE | ~800-810 GB | **~400-430 GB** |

*Note 1: Vision-Language models require additional VRAM for vision encoders and high-resolution image context. The estimation includes basic KV Cache, but we recommend reserving an extra 2-4 GB for complex vision tasks.*
*Note 2: For MoE models, all expert weights must be loaded into memory simultaneously. Therefore, their VRAM footprint depends on the total parameter count, not just the activated parameters.*

1. Setup Python Environment
```bash
conda create -n chemeagle python=3.10
conda activate chemeagle
pip install -r requirements.txt
```

2. Download the necessary [models](https://huggingface.co/CYF200127/ChemEAGLEModel/tree/main) and put in the main path.

3. Deploy vLLM Server

**For Linux:**
```
conda create -n vllm_env python=3.10
conda activate vllm_env
pip install vllm
vllm serve /path/to/Qwen3-VL-32B-Instruct \
    --port 8000 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 64000 \
    --limit-mm-per-prompt video=0
conda activate chemeagle
```

**For Windows (PowerShell):**

```powershell
docker run -d --gpus all `
    -p 8000:8000 `
    -v /path/to/Qwen3-VL-32B-Instruct:/models/Qwen3-VL-32B-Instruct `
    --name vllm-server `
    vllm/vllm-openai:latest `
    --model /models/Qwen3-VL-32B-Instruct `
    --port 8000 `
    --trust-remote-code `
    --enable-auto-tool-choice `
    --tool-call-parser hermes `
    --max-model-len 64000 `
    --limit-mm-per-prompt.video 0
```

**Note:** 
- Replace `/path/to/Qwen3-VL-32B-Instruct` with your actual model path. For example `/models/Qwen3-VL-32B-Instruct`.
- The vLLM server will be available at `http://localhost:8000/v1` by default.



4. After the vLLM server is running, you can use the open source version of ChemEAGLE as follows:

```python
from main import ChemEagle_OS

# Using default local vLLM server (http://localhost:8000/v1)
image_path = './examples/1.png'
results = ChemEagle_OS(image_path)
print(results)
```

5. Alternatively, run the following code to extract machine-readable chemical data from chemical literature (PDF files) directly:
```python
import os
from main import ChemEagle_OS
from pdf_extraction import run_pdf
pdf_path   = 'your/pdf/path'
output_dir = 'your/output/dir'
run_pdf(pdf_dir=pdf_path, image_dir=output_dir)
results = []
for fname in sorted(os.listdir(output_dir)):
    if not fname.lower().endswith('.png'):
        continue
    img_path = os.path.join(output_dir, fname)
    try:
        r = ChemEagle_OS(img_path)
        r['image_name'] = fname
        results.append(r)
    except Exception as e:
        results.append({'image_name': fname, 'error': str(e)})
print(results)
```

## 🗂️ Benchmarking
All benchmark datasets and ground truth can be found in our [Huggingface Repo](https://huggingface.co/datasets/CYF200127/ChemEagle/tree/main).

## 🌐 Chemical information extraction using [ChemEAGLE.Web](https://app.chemeagle.net/) 

[comment]: <> ()
![visualization](examples/webapp2.png)
<div align="center"> The interface of ChemEAGLE.Web. </div>
  

Go to our [ChemEAGLE.Web app demo](https://app.chemeagle.net/) to directly use our tool online for both image and PDF input! The built-in Ketcher editor on the right lets you instantly visualize, verify, and fine-tune any extracted molecular structure before exporting. Feel free to provide us with any feedback too! (Note: The app runs on the HPC4.ust.hk server with a maximum uptime of 3 days; it is restarted for maintenance every three days, please wait a moment if the site is temporarily unavailable or raises a connection error.)

#### May 30 Update: Fixed some output bugs.

### 🐍 Programmatic access: ChemEAGLE API & Python SDK (Testing)

Don't want to install anything? The same engine that powers
[`app.chemeagle.net`](https://app.chemeagle.net/) is also exposed as an
authenticated JSON API at `https://app.chemeagle.net/api/v1`. The easiest
way to use it is the **Python SDK** (1 file, only depends on `requests`).

#### Install

```bash
# Option A: pip install from the api/sdk subfolder of this repo
pip install -e sdk

# Option B: zero-install — just copy sdk/chemeagle_client/client.py
#          into your project (only depends on `requests`).
```

#### Get an API key

Ask the maintainers (or open a GitHub issue) for a key. Keys look like
`ce_xxxxxxxxxxxxxxxxxxxxxxx`. Set it via env var so you don't have to
hard-code it:

```bash
export CHEMEAGLE_API_KEY="ce_xxxxxxxxxxxxxxxxxxxxxxx"
```

#### Example Usage

```python
from chemeagle_client import ChemEagleClient

client = ChemEagleClient(base_url="https://app.chemeagle.net")  # picks up env key

# 1) Single image (synchronous — easiest)
result = client.process_image("examples/1.png", sync=True)
for rxn in result.get("reactions", []):
    print(rxn["id"], "→", rxn["smiles"])

# 2) PDF (asynchronous — recommended, PDFs take minutes)
task_id = client.process_pdf("paper.pdf")["task_id"]
final = client.wait_for_task(task_id, poll=3.0, max_wait=1800)
for img in final["results"]:
    print(img["image_name"], len(img.get("reactions", [])), "reactions")

# 3) From a public URL (server downloads on your behalf)
client.process_url("https://example.org/paper.pdf", kind="pdf", sync=False)
```

Errors raise `ChemEagleError` with a stable `error_code` (`rate_limited`,
`invalid_api_key`, `payload_too_large`, …) — easy to handle and retry.

#### Full reference

- 📖 **Live docs**: <https://app.chemeagle.net/api/v1/docs>


When the input is a multimodal chemical reaction graphic:
![visualization](examples/reaction9.png)
<div align="center",width="100">
</div> 

The output dictionary should be a complete machine-readable reaction list with reactant SMILES, product SMILES, detailed conditions and additional information for every reaction in the graphics. For example:

``` 
{"reactions":[
{"reaction_id":"0_1","reactants":[{"smiles":"[Ar]C([R])=C=O","label":"1"},{"smiles":"Cc1ccc(S(=O)(=O)N2OC2c2ccccc2Cl)cc1","label":"2"}],
"conditions":[{"role":"reagent","text":"10 mol% B17 or B27","smiles":"C(C=CC=C1)=C1C[N+]2=CN3[C@H](C(C4=CC=CC=C4)(C5=CC=CC=C5)O[Si](C)(C)C(C)(C)C)CCC3=N2.F[B-](F)(F)F","label":"B17"},
{"role":"reagent","text":"10 mol% B17 or B27","smiles":"CC(C)C(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F","label":"B27"},
{"role":"reagent","text":"10 mol% Cs2CO3","smiles":"[Cs+].[Cs+].[O-]C(=O)[O-]"},{"role":"solvent","text":"PhMe","smiles":"Cc1ccccc1"},{"role":"temperature","text":"rt"},{"role":"yield","text":"38 - 78%"}],
"products":[{"smiles":"[Ar]C1([R])O[C@H](c2ccccc2Cl)N(S(=O)(=O)c2ccc(C)cc2)C1=O","label":"3"}]},

{"reaction_id":"1_1","reactants":[{"smiles":"CCC(=C=O)c1ccccc1","label":"1a"},{"smiles":"Cc1ccc(S(=O)(=O)N2OC2c2ccccc2Cl)cc1","label":"2a"}],
"conditions":[{"role":"reagent","text":"10 mol% B17 or B27","smiles":"C(C=CC=C1)=C1C[N+]2=CN3[C@H](C(C4=CC=CC=C4)(C5=CC=CC=C5)O[Si](C)(C)C(C)(C)C)CCC3=N2.F[B-](F)(F)F","label":"B17"},
{"role":"reagent","text":"10 mol% B17 or B27","smiles":"CC(C)C(C=CC=C1)=C1[N+]2=CN3[C@H](C(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))(C1=CC(=CC(=C1C(F)(F)F)C(F)(F)F))O)CCC3=N2.F[B-](F)(F)F","label":"B27"},
{"role":"reagent","text":"10 mol% Cs2CO3","smiles":"[Cs+].[Cs+].[O-]C(=O)[O-]"},{"role":"solvent","text":"PhMe","smiles":"Cc1ccccc1"},{"role":"temperature","text":"rt"},{"role":"yield","text":"71%"}],
"products":[{"smiles":"CC[C@@]1(c2ccccc2)O[C@H](c2ccccc2Cl)N(S(=O)(=O)c2ccc(C)cc2)C1=O","label":"3a"}],"additional_info":[{"text":"14:1 dr, 91% ee"}]},

{"reaction_id":"2_1",... ###More detailed reactions}
],
"text_extraction":"..."
}
```
Note: Due to the characteristics of LLMs, slight variations in outputs may occur.
The input can be any chemical graphics; feel free to try more examples! 


![visualization](examples/reaction5.png)
![visualization](examples/reaction1.jpg)
![visualization](examples/reaction2.png)
![visualization](examples/reaction4.png)
![visualization](examples/reaction6.png)
![visualization](examples/molecules1.png)

## :warning: Acknowledgement
1. We use api_version="2024-10-21" with the HKUST Azure OpenAI endpoint as our official closed-source version.
2. Our code is based on [MolNexTR](https://github.com/CYF2000127/MolNexTR), [MolScribe](https://github.com/thomas0809/MolScribe), [RxnIM](https://github.com/CYF2000127/RxnIM), [RxnScribe](https://github.com/thomas0809/RxNScribe), [ChemNER](https://github.com/Ozymandias314/ChemIENER), [ChemRxnExtractor](https://github.com/jiangfeng1124/ChemRxnExtractor), [AutoAgents](https://github.com/Link-AGI/AutoAgents), and [Azure OpenAI](https://azure.microsoft.com/).


