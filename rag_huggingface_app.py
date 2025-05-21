# rag_hf_local.py

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="HF RAG Chat", layout="wide")
st.title("📄 RAG Chat Using Hugging Face (Local Transformers)")

# Upload files
uploaded_files = st.file_uploader("Upload PDFs, TXT, or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# Ask question
query = st.text_input("Ask a question about your documents:")

# Function: Load and parse documents
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

# Run pipeline when user uploads and asks
if uploaded_files and query:
    with st.spinner("🔍 Processing..."):
        # 1. Load and chunk documents
        raw_docs = load_documents(uploaded_files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(raw_docs)

        # 2. Embed documents using HuggingFace sentence transformer
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # 3. Load local Hugging Face model (via transformers)
        generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=512)
        llm = HuggingFacePipeline(pipeline=generator)

        # 4. Run RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain({"query": query})

        # 5. Show answer
        st.subheader("💬 Answer")
        st.write(result["result"])

        # 6. Show source document snippets
        st.subheader("📚 Source Documents")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Snippet {i+1}:** `{doc.metadata.get('source', 'Uploaded File')}`")
            st.write(doc.page_content[:300] + "...")

