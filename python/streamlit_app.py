from dotenv import load_dotenv
load_dotenv()

from retriever import JSONRetriever
import streamlit as st
from streamlit import chat_message
import os

st.header("Thoughtful AI Customer Support Chatbot")

st.session_state.script_dir = os.path.dirname(os.path.abspath(__file__))
st.session_state.retriever = JSONRetriever(collection_name="customer_support", chroma_persist_directory=os.path.join(st.session_state.script_dir, "./chroma_db"))

if "user_prompt_history" not in st.session_state:
    st.session_state.user_prompt_history = []

if "response_history" not in st.session_state:
    st.session_state.response_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.text_input("Prompt", placeholder="Ask anything about Thoughtful AI")

if prompt:
    with st.spinner("Generating response..."):
        
        # the return_context field can be used to return the source documents to the user. but to do this the JSON metadata needs to be updated to include the source field.
        answer = st.session_state.retriever.chat(prompt, return_context=False, chat_history=st.session_state.chat_history)
        
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.response_history.append(answer)
        st.session_state.chat_history.append(("human", prompt))
        st.session_state.chat_history.append(("assistant", answer))    
        
if st.session_state.chat_history:   
    for response, query in zip(st.session_state.response_history, st.session_state.user_prompt_history): 
        chat_message("user").write(query)
        chat_message("assistant").write(response)