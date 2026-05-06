from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_docs():
    loader = DirectoryLoader(
        "./data/pdf",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    return loader.load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

def main():
    print("Loading PDFs...")
    docs = load_docs()

    print("Docs loaded:", len(docs))

    if len(docs) == 0:
        print("❌ No PDFs found. Check your path!")
        return

    print("Splitting documents...")
    chunks = split_docs(docs)

    print("Chunks created:", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("faiss_index")

    print("✅ FAISS index created and saved!")

if __name__ == "__main__":
    main()