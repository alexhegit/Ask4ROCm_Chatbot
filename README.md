# Ask4ROCm_Chatbot
Learn ROCm with chatbot which powered by AMD ROCm solution.

## Demo Show
![](./resources/Ask4ROCm_Chatbot_Demo.gif)


## Supported Hardware
Any AMD GPUs supported by ROCm should work for this Chatbot. You may find the GPU list from [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

- AMD CDNA GPU:  MI300 / MI200 / MI100, etc
- AMD RDNA GPU: Radeon 7000 series / Radeon  6000 series / iGPU 780M ,etc
- AMD CPU (w/o ROCm)

**NOTE**

AMD iGPU 780M is a very powerful integreated GPU of AMD Ryzne GPU. Please refer to https://github.com/alexhegit/Playing-with-ROCm/blob/main/inference/LLM/Run_Ollama_with_AMD_iGPU780M-QuickStart.md to enable AMD iGPU-780M with ROCm.

Certenly, you could use any AMD CPU in your hand with Ollama to do the LLM inference if the system w/o any AMD GPU or NVIDIA GPU. 


## Software Installation
This chatbot depends on many OSS projects.
- Ubuntu OS
- AMD ROCm
- WebUI: Streamlit
- RAG pipeline: LlamaIndex
- VectorDB: ChromaDB
- LLM inference enginee: Ollama
  - LLM: Llama3-7b/tinyllama or others
  - Embedding Model: nomic-embed-text

### ROCm
Refer to https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html to install the ROCm components.

Then to setup a Python base environment to run the chatbot application. You may use conda or python venv to manage it.

### Streamlit
pip install streamlit

### LlamaIndex
pip install llama-index
pip install llama-index-core llama-index-readers-file 
pip install llama-index-llms-ollama llama-index-embeddings-ollama
pip install llama-index-vector-stores-chroma

### Ollama
Please install Ollama refer to https://ollama.com/download

curl -fsSL https://ollama.com/install.sh | sh

Then download the models as reqerired.
ollama pull llama3
ollama pull tinyllama
ollama pull nomic-embed-text

Please use `pip install -r requirements.txt` for easy installation.

--------------------------------------------------------------------------------------------------
### PyTorch_ROCm
This Apps does not depend on PyTorch at NOW. But we suggest to install the PyTorch-rocm for further work.

Install PyTorch-rocm from https://pytorch.org/
e.g. on linux
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0


## Appendix
If you is a beginer of ROCm ,LLamaIndex and Ollama. Here are other repos may help you to learn and hands-on with them.
1. Step by Step from labs in jupyter-notebook to webui demo for creating the RAG apps with ROCm+LLamaIndex+Ollama:   
   https://github.com/alexhegit/RAG_LLM_QnA_Assistant
   
1. Misc hands-on of ROCm:
   https://github.com/alexhegit/Playing-with-ROCm


```
@misc{AlexTryMachineLearning,
  author =   {He Ye (Alex)},
  title =    {Ask4ROCm_Chatbot: assist to learn ROCm with RAG},
  howpublished = {\url{https://alexhegit.github.io/}},
  year = {2024--}
}
```
