## Resoloute_AI_TASK4
# 📄 RAG Chat App using Hugging Face (Streamlit)

This project is part of Resolute AI's Internship Assignment – **Task 4: RAG (Retrieval-Augmented Generation)**.

It is a Streamlit web application that allows users to:

- ✅ Upload multiple documents (PDF, TXT, DOCX)
- ✅ Ask natural language questions about the content
- ✅ Get intelligent answers using a lightweight Hugging Face model (FLAN-T5)

---

## 🚀 Features

- 🧠 **Document Parsing**: Loads and reads PDF, TXT, and DOCX files using LangChain loaders.
- 🔍 **Semantic Search**: Splits and embeds the documents into chunks using `sentence-transformers` and stores them in FAISS.
- 💬 **Question Answering**: Retrieves relevant text chunks and passes them to a Hugging Face model (`google/flan-t5-small`) for response generation.
- 🌐 **Web UI**: Built with Streamlit and deployed on [Streamlit Cloud](https://streamlit.io/cloud) for public access.

---

## 🛠️ Tech Stack

| Layer             | Tool/Library                                      |
|------------------|----------------------------------------------------|
| UI               | Streamlit                                          |
| Embeddings       | sentence-transformers / all-MiniLM-L6-v2           |
| Vector DB        | FAISS                                              |
| LLM              | Hugging Face (FLAN-T5-small) via `transformers`    |
| NLP Framework    | LangChain + langchain-community                    |

---

## 📦 Installation (Local Setup)

1. Clone this repo:
```bash
git clone https://github.com/nita70/Resolute_AI_TASK4.git
cd Resolute_AI_TASK4
