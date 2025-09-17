# src/chains/rag_chain.py
from langchain.chains import RetrievalQA
from src.llm.model import get_llm

def run_rag(query, vectorstore, model_name="llama3"):
    llm = get_llm(model_name)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )
    return qa.run(query)
