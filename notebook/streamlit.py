import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------
# Load embeddings + FAISS
# -------------------------
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = load_retriever()

# -------------------------
# UI
# -------------------------
st.title("📄 RAG PDF Chat")

query = st.text_input("Ask something from your PDFs:")

if query:
    docs = retriever.invoke(query)

    st.subheader("Top Results:")

    for i, doc in enumerate(docs):
        st.write(f"**Result {i+1}:**")
        st.write(doc.page_content[:300])
        st.write(f"Source: {doc.metadata.get('file_name')}")
        st.write("---")