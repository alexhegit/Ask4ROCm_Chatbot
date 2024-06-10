# Ask4ROCm_Chatbot
Learn ROCm with chatbot which powered by AMD ROCm solution.

## Supported hardware
- AMD CDNA GPU: MI200 / MI300
- AMD RDNA GPU: Radeon 7000 series / Radeon  6000 series / iGPU 780M

**NOTE**

Please refer to https://github.com/alexhegit/Playing-with-ROCm/blob/main/inference/LLM/Run_Ollama_with_AMD_iGPU780M-QuickStart.md to enable AMD iGPU-780M with ROCm.


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






- 
