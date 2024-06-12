# Lab7 Use streamlit to run RAG_LLM Q&A

# Usage: At the dir of it and run the command "streamlit run ./Lab7-LlamaIndex_CUM_ST.py"

# Question Examples
"""
What is MIGraphX?

How to install MIGraphX?
"""

import os
# Enalbe 780M with ROCm
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import time
import pathlib
import shutil
import streamlit as st

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
#from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

st.set_page_config(page_title="Your Local Chatbot, assist to learn ROCm", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
#st.info("Powered by ROCm & LlamaIndex 💬🦙")
#st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
st.title("Learn ROCm with Chatbot \n powered by AMD!")
st.image("https://www.amd.com/content/dam/amd/en/images/logos/products/amd-rocm-lockup-banner.jpg")
         

# Create Service Context
#@st.cache_resource(show_spinner=False)
def create_ServiceContext(llm_name):
    # Set embedding model
    # Please download it ahead running this lab by "ollama pull nomic-embed-text"
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Set ollama model
    Settings.llm = Ollama(model=llm_name, request_timeout=160.0, temperature=llm_temperature)

    if "service_context" not in st.session_state.keys():
        st.session_state.service_context = ServiceContext.from_defaults(llm=Settings.llm,
                                                               embed_model=Settings.embed_model,
                                                               system_prompt="You are an expert on AMD ROCm and your job is to answer technical questions. Assume that all questions are related to the documentation of ROCm. Keep your answers technical and based on facts – do not hallucinate features.")

    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about AMD ROCm!"}
        ]
    return

def buid_index(service_context, dbpath):
    # initialize client
    st.session_state.db = chromadb.PersistentClient(
        path=dbpath,
        #settings=Settings(allow_reset="True"),
        #tenant=DEFAULT_TENANT,
        #database=DEFAULT_DATABASE,
    )
    # get collection
    chroma_collection = st.session_state.db.get_or_create_collection(st.session_state["db_collection"])
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Load data
    docs = SimpleDirectoryReader(input_dir=save_folder).load_data()
    # Build vector index per-document
    index = VectorStoreIndex.from_documents(
        docs,
        show_progress=True,
        service_context=service_context, 
        storage_context=storage_context,
        transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=50)],
    )
    return index

def load_index(service_context, dbpath):
    # initialize client
    st.session_state.db = chromadb.PersistentClient(
        path=dbpath,
        #settings=Settings(allow_reset="True"),
        #tenant=DEFAULT_TENANT,
        #database=DEFAULT_DATABASE,
    )
    # get collection
    #chroma_collection = st.session_state.db.get_or_create_collection(st.session_state["db_collection"])
    chroma_collection = st.session_state.db.get_collection(st.session_state["db_collection"])

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context, 
        storage_context=storage_context
    )
    return index


# Setting in sidebar
st.sidebar.header("Model")
llm_name=st.sidebar.selectbox("", ("llama3", "tinyllama"))
llm_temperature = st.sidebar.slider('Temperature', 0.0, 1.0, 0.6, step=0.01,)

if "config_init" not in st.session_state:
    st.session_state["config_init"] = True
    st.session_state["reindex"] = False
    st.session_state["db_collection"] = "db_collection"

# Add an "upload file" button
st.sidebar.header("RAG Reference File")
save_folder = "./data"
if not os.path.exists(save_folder):
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
uploaded_file = st.sidebar.file_uploader(label="")
# Check if a file has been uploaded
if uploaded_file is not None:
    save_path = pathlib.Path(save_folder, uploaded_file.name)
    with open(save_path, mode="wb") as w:
        w.write(uploaded_file.getvalue())
        st.session_state['reindex'] = True

# ReIndex by cache.clear
#st.sidebar.header("Clear Cache")
if st.sidebar.button("ReIndex"):
    #shutil.rmtree('./chroma_db', ignore_errors=True)
    st.session_state['reindex'] = True
    st.cache_resource.clear()
st.sidebar.markdown("NOTE: More time for rebuilding the Index!")

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the ROCm docs – hang tight! This should take 2-3 minutes."):
        service_context = st.session_state.service_context
        if ((not os.path.exists("./chroma_db/ROCm_db")) or (st.session_state['reindex'] == True)):
            index = buid_index(service_context, dbpath="./chroma_db/ROCm_db")
            st.session_state['rebuild_index'] = False
        else:
            index = load_index(service_context=service_context, dbpath="./chroma_db/ROCm_db")
 
        return index

st.write("Selected Model", llm_name)
create_ServiceContext(llm_name)

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, streaming=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response.response_gen)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
