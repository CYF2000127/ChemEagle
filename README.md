# ChemEagle


## :sparkles: Highlights
<p align="justify">
In this work, we present ChemEagle, a multimodal large language model (MLLM)-based multi-agent system that integrates diverse chemical information extraction tools to extract multimodal chemical reactions. By integrating 7 expert-designed tools and 6 chemical information extraction agents, ChemEagle not only processes individual modalities but also utilizes MLLMs' reasoning capabilities to unify extracted data, ensuring more accurate and comprehensive reaction representations. By bridging multimodal gaps, our approach significantly improves automated chemical knowledge extraction, facilitating more robust AI-driven chemical research.

[comment]: <> ()
![visualization](figure/arch.png)
<div align="center">
An example workflow of our ChemEagle. It illustrates how ChemEagle extracts and structures multimodal chemical reaction data. Each agent handles specific tasks, from reaction image parsing and molecular recognition to SMILES reconstruction and condition role interpretation, ensuring accurate and structured data integration.
</div> 

## 🔥 Using the code and the model
### Using the code
Clone the following repositories:
```
git clone https://github.com/CYF2000127/ChemEagle
```
### Example usage of the model
1. First create and activate a [conda](https://numdifftools.readthedocs.io/en/stable/how-to/create_virtual_env_with_conda.html) environment with the following command in a Linux, Windows, or MacOS environment (Linux is the most recommended):
```
conda create -n chemeagle python=3.10
conda activate chemeagle
```

2. Then Install requirements:
```
pip install -r requirements.txt
```

3. Set up your API keys in your environment.or add your api key in the [api_key.txt](./api_key.txt)
```
export API_KEY=your-openai-api-key
```


4. Run the following code to extract multimodal chemical reactions from a multimodal reaction image:
```python
from main import ChemEagle
image = './examples/1.png'
results = ChemEagle(image_path)
print(results)
