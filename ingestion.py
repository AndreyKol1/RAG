from transformers import pipeline
import streamlit as st
ner = pipeline("ner", "dslim/bert-large-NER", tokenizer="dslim/bert-large-NER", aggregation_strategy="simple") # initializing BERT for NER (additional check for relevance apart from keyboard)


def get_text_from_doc(files): 
    docs = []
    for document in files: # looping through all documents and converting them to strings and storing them in a list
        if document.type == "text/plain":
            file_content = document.read().decode("utf-8")
            docs.append(file_content)
        else:
            st.warning(f"Unsupported file type: {document.type}")
    return docs

def check_relevance(documents):
    valid_documents = []
    
    CONCERT_KEYWORDS = [
        "concert", "tour", "venue", "performer", "setlist",
        "stage", "backstage", "tickets", "opening act", "tour bus",
        "logistics", "soundcheck", "dates", "gig", "headliner", "arena", "date"
        ] # basic keywords that will be checked in the document
    
    for i, text in enumerate(documents):
        res = ner(text)
        entities = [ent['word'].lower() for ent in res  if ent['entity_group'] in ["PER", "LOC", "ORG", "MISC"]]  # NER as additional check
        
        relevant_words = [k for k in CONCERT_KEYWORDS if k in text.lower()] # checking basic keywords
                
        if len(entities) > 1 and len(relevant_words) > 1: # simple checking stage
            valid_documents.append(text)
            
    return True if len(valid_documents) == len(documents) else False


