import tempfile
import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain_core.documents import Document

import pandas as pd
import pdfplumber

from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

st.set_page_config(page_title="RAG-chatbot", layout="wide")
st.title("CHAT WITH YOUR FILE")
uploaded_file = st.file_uploader(
    "Upload a file",
    type=["pdf", "docx", "csv"]
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

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            # PDF
            if file_extension == ".pdf":

                text = ""

                loader = PyPDFLoader(temp_path)
                pdf_docs = loader.load()

                for doc in pdf_docs:
                    text += doc.page_content + "\n"

                # Extract tables separately
                with pdfplumber.open(temp_path) as pdf:

                    for page in pdf.pages:

                        tables = page.extract_tables()

                        for table in tables:

                            for row in table:

                                cleaned_row = [
                                    str(cell) if cell else ""
                                    for cell in row
                                ]

                                text += " | ".join(cleaned_row) + "\n"

                documents = [Document(page_content=text)]

            # DOCX
            elif file_extension == ".docx":

                doc = DocxDocument(temp_path)

                text = ""

                # Normal paragraphs
                for para in doc.paragraphs:
                    text += para.text + "\n"

                # Tables
                for table in doc.tables:

                    for row in table.rows:

                        row_text = [
                            cell.text for cell in row.cells
                        ]

                        text += " | ".join(row_text) + "\n"

                documents = [Document(page_content=text)]

            # CSV
            elif file_extension == ".csv":

                df = pd.read_csv(temp_path)

                text = df.to_string(index=False)

                documents = [Document(page_content=text)]

            else:

                st.error("Unsupported file type")
                st.stop()

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