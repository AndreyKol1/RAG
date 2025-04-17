from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def add_doc(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # initializing model for vector embedding, which will allow similarity-search
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings, collection_name="rag_collection") # store data in database 
    return vectorstore


