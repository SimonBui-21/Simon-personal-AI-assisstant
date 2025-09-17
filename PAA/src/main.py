# src/main.py
from src.embeddings.embedder import build_vectorstore
from src.chains.rag_chain import run_rag
from src.config import OLLAMA_MODEL


def main():
    print("🤖 Personal AI Assistant (Ollama Edition)")
    query = input("Ask me something: ")

    vectorstore = build_vectorstore("data/my_notes.txt")
    answer = run_rag(query, vectorstore, model_name=OLLAMA_MODEL)

    print("\nAssistant:", answer)

if __name__ == "__main__":
    main()
