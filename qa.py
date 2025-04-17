from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

def get_conversation_chain(vectorstore):
    pipe = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    ) # initializing model that will generate outputs
     
    model = HuggingFacePipeline(pipeline=pipe)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) # memory in which conversation is stored
    
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    ) # Creating a conversational RAG pipeline
    return conversation_chain
