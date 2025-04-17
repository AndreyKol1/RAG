from transformers import pipeline

summarizer = pipeline("summarization", "philschmid/bart-large-cnn-samsum", tokenizer="philschmid/bart-large-cnn-samsum") # initializing model for summarization

def summarize(docs):
    return [summarizer(doc) for doc in docs] # making a summary of each document
        