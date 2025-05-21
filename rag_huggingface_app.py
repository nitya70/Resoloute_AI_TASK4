# rag_huggingface_app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
import tempfile

# 🔐 Hugging Face API Key (can be hardcoded or input)
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#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
if not HF_API_KEY:
    st.warning("Please enter your Hugging Face API key.")
    HF_API_KEY = st.text_input("Hugging Face API Key", type="password")
    if HF_API_KEY:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY

st.set_page_config(page_title="HF RAG Chat", layout="wide")
st.title("🤗 RAG Chat using Hugging Face")

uploaded_files = st.file_uploader("Upload PDFs, DOCX, or TXT", type=["pdf", "txt", "docx"], accept_multiple_files=True)
query = st.text_input("Ask a question about your documents:")

# Load and parse documents
def load_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.error(f"Unsupported file format: {file.name}")
            continue

        docs.extend(loader.load())
    return docs

# Run RAG when user uploads and asks
if uploaded_files and query:
    with st.spinner("Processing..."):
        raw_docs = load_documents(uploaded_files)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(raw_docs)

        # Embeddings via Hugging Face
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # Hugging Face LLM (Mistral or Falcon hosted)
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # or try 'tiiuae/falcon-7b-instruct'
            model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain({"query": query})

        st.subheader("💬 Answer")
        st.write(result["result"])

        st.subheader("📚 Sources")
        for doc in result["source_documents"]:
            st.markdown(f"- `{doc.metadata.get('source', 'Unknown')}`")
