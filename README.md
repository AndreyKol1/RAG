# RAG
**Document Relevance Checker & Question Answering System**

## Project Information

**RAG concert** is a lightweight tool designed to evaluate the relevance of uploaded documents to a user-specified query using Retrieval-Augmented Generation (RAG). It combines NER-based preprocessing, keyword filtering, document summarization, and large language models (LLMs) to provide meaningful answers and insights—all through an easy-to-use web interface.

---

## Features

- Upload and process documents in one click  
- Named Entity Recognition and keyword-based relevance filtering  
- Automatic document summarization  
- Vector storage with ChromaDB  
- Question answering using a lightweight LLM (Flan-T5)  
- Optional web search using SerpAPI  
- Intuitive UI for a seamless user experience

---

## Installation

It’s recommended to use a virtual environment. After activating it, install all dependencies with:

```bash
pip install -r requirements.txt
```

## API Keys
This project requires API keys for:

Hugging Face (for models like bert-large, bart-large, and flan-t5-base)

SerpAPI (optional, for web search)

### Setup:
Create a token at Hugging Face Tokens with read access.

Get a token at SerpAPI.

Create a .env file in the project directory with the following:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
SERPAPI=your_serpapi_token

## Usage 

To lauch the app run a command below

```bash
streamlit run app.py
```

Once the UI loads, you can:

1. Enter a question related to a topic of interest.
2. Upload one or more documents.

## Tech stack

1. Frontend: Streamlit
2. Backend: FastAPI + Python
3. NER: bert-large
4. Summarization: bart-large
5. LLM (QA): flan-t5-base
6. Vector Store: ChromaDB
7. Search (Optional): SerpAPI






