import tempfile
import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="RAG-chatbot", layout="wide")
st.title("chat with your PDFs")
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

current_dir = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(current_dir, "faiss_index")

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = load_resources()

with st.sidebar:

    if uploaded_file is not None:

        if st.button("Create Index"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            loader = PyPDFLoader(temp_path)

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(documents)

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            vectorstore.save_local(INDEX_PATH)

            st.success("Index Created!")

            os.remove(temp_path)


if os.path.exists(INDEX_PATH):
    
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    query = st.text_input("Ask a question:")
    if query:
        
        docs = vectorstore.similarity_search(query, k=6)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using only the context below:

        {context}

        Question: {query}
        """

        llm = Ollama(model="phi3:mini")
        response = llm.invoke(prompt)

        st.write("### Answer:")
        st.write(response)
else:
    st.info("Please click 'Rebuild Index' in the sidebar.")