import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="RAG Fix", layout="wide")
st.title("🚀 Mark Zuckerberg Doc Search")

# PATH CORRECTION based on your ls -R
# Your PDF is at: ./data/pdf/markzuck.pdf
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, "data", "pdf")
INDEX_PATH = os.path.join(current_dir, "faiss_index")

@st.cache_resource
def load_resources():
    # Use the model defined in your source[cite: 1]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = load_resources()

# Sidebar Indexing
with st.sidebar:
    if st.button("Rebuild Index"):
        all_docs = []
        if os.path.exists(DATA_PATH):
            pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
            for pdf in pdf_files:
                loader = PyPDFLoader(os.path.join(DATA_PATH, pdf))[cite: 1]
                all_docs.extend(loader.load())
            
            # Split logic[cite: 1]
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = splitter.split_documents(all_docs)
            
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(INDEX_PATH)
            st.success("Index Updated!")
        else:
            st.error(f"Path not found: {DATA_PATH}")

# Search Interface
if os.path.exists(INDEX_PATH):
    # Load existing index[cite: 1]
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    query = st.text_input("Ask a question:")
    if query:
        # Retrieve results[cite: 1]
        if query:

            docs = vectorstore.similarity_search(query, k=3)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
            Answer the question using only the context below:

            {context}

            Question: {query}
            """

            llm = Ollama(model="llama3")
            response = llm.invoke(prompt)

            st.write("### Answer:")
            st.write(response)
        else:
            st.info("Please click 'Rebuild Index' in the sidebar.")