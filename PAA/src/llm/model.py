# src/llm/model.py
from langchain_community.llms import Ollama

def get_llm(model_name="llama3"):
    return Ollama(model=model_name)
