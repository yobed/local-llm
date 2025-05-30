import streamlit as st
from ollama import Client

# client
client = Client(host="http://localhost:11434")

# yield chunks (for streaming responses)
def ollama_stream(history):
    for chunk in client.chat(
            model="deepseek-r1:7b",
            messages=history,
            stream=True):
        yield chunk["message"]["content"] 


st.title("Local LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Type a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(ollama_stream(st.session_state.messages))

    st.session_state.messages.append({"role": "assistant", "content": response})
