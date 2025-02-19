import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat/generate"

st.title("MiniGPT Chatbot")
st.write("Ask a question and see how MiniGPT responds!")

user_input = st.text_input("Enter your question:", "")

if st.button("Send"):
    if user_input:
        response = requests.post(API_URL, json={"prompt": user_input, "max_length": 20})

        if response.status_code == 200:
            st.write("### MiniGPT's Response:")
            st.write(response.json()["response"])
        else:
            st.error("Error communicating with the API.")
    else:
        st.warning("Please enter a question before sending.")