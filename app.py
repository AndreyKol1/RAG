import streamlit as st
import torch
from ingestion import get_text_from_doc, check_relevance
from summarization import summarize
from vector_db import add_doc
from qa import get_conversation_chain
from search import look_for_artists, extract_tour_data
from langchain_text_splitters.character import CharacterTextSplitter

torch.classes.__path__ = [] # this line helped fix an error of not finding path

# function to chunk text. This ensures better contextual understanding, helps RAG capture relationships
def chunk_text(summaries, chunk_size=500, overlap=100):
    chunks = []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=overlap, length_function=len)
    for i, _ in enumerate(summaries):
            chunks.extend(splitter.split_text(summaries[i][0]["summary_text"]))
    
    return chunks

# function which performs online search for user provided artist or band
def online_search(user_question):
    with st.spinner("Searching at Tickermaster for events..."):
        results = look_for_artists(user_question)
        events = extract_tour_data(results)
            
        if not events:
            print("No upcoming events")
        else:
            st.subheader("Upcoming events:")
            for i, event in enumerate(events, start=1):
                st.markdown(f"Event {i}: {event['title']}")
                st.markdown(f"Link: {event['link']}")
                st.markdown(f"Date: {event['date']}")
                st.markdown(f"Location: {event['location']}")

# main function with ui and additional logic
def main():

    st.set_page_config(page_title="Load your documents",
                       page_icon=":document:") # title
    

    if "conversation" not in st.session_state:
        st.session_state.conversation = None # declare conversation state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None # declare chat history state

    st.header("Load your documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation: # searching only if question is asked and llm initialized
        with st.spinner("searching for an answer..."):
            result = st.session_state.conversation.invoke({"question": user_question}) # give questing to the model

            if st.session_state.chat_history is None: 
                st.session_state.chat_history = [] # creating a list to store conversation

            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"]}) # appending bot answer
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}) # appending user query
                
            for message in st.session_state.chat_history[::-1]: # starting from the end because user query was added lastly to simulate logical conversations. New messages appear at the top, not at the bottom of the conversation
                if message["role"] == "user":
                    st.write(f"🧑 You: {message['content']}")
                else:
                    st.write(f"🤖 Bot: {message['content']}")
                
    summary = None
    document = None

    with st.sidebar: # initializing side bar for adding documents
        st.subheader("Your documents")
        document = st.file_uploader(
            "Upload your plain text file here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"): # creating a spinner to wait until all processes are done
                raw_text = get_text_from_doc(document) # getting text from document
                relevance = check_relevance(raw_text) # checking documents for relevance according to the rules of the task
                if relevance: # all document pass checks
                    summary = summarize(raw_text) # getting summaries from out documents
                                
                    text_chunks = chunk_text(summary) # dividing our documents into chunks

                    vector_store = add_doc(text_chunks) # storing documents in ChromaDB

                    st.session_state.conversation = get_conversation_chain(vector_store) # creates and stores conversation pipeline
                    st.success("Document successfully processed and added to database.")
                else:
                    st.write("Sorry, I cannot ingest documents with other themes.")
    if summary: # printing summary
        st.subheader("Summaries of uploaded documents")
        for i, item in enumerate(summary):
            with st.expander(f"Document {i+1} summary"):
                st.write(item[0]["summary_text"])
                
    if user_question and not document: 
        online_search(user_question) # performing online search 
            
if __name__ == '__main__':
    main()