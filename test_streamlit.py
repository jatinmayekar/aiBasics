import openai
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openai import OpenAI
import os
import sys
import json
import requests
import time
import datetime
import random
import re

from io import StringIO

import load_dotenv
from load_dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPEN_API_KEY")
client = OpenAI()

st.title("OpenAI Bot")

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")   
    st.button("Analyze", type="primary")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    file_name = uploaded_file.name
    print(file_name)
    
    with open('swe_exhibitor_file.pdf', 'wb') as f:
        f.write(bytes_data)
    st.success("PDF file saved successfully.")

    file_1 = client.files.create(
        file=open("swe_exhibitor_file.pdf", "rb"),
        purpose='assistants'
    )
    assistant = client.beta.assistants.create(
        name="Exhibitor Assistant",
        instructions="You are an greeter at the SWE 2023 conference. You have documents with information about the exhibitors."
                    "Use your knowledgre retrieval skills to answer questions about the exhibitors.",
        tools=[{"type": "code_interpreter"},{"type":"retrieval"}],
        model="gpt-4-1106-preview",
        file_ids=[file_1.id]
    )
elif uploaded_file is None:
    assistant = client.beta.assistants.create(
        name="Exhibitor Assistant",
        instructions="You are an greeter at the SWE 2023 conference. You have documents with information about the exhibitors."
                    "Use your knowledgre retrieval skills to answer questions about the exhibitors.",
        tools=[{"type": "code_interpreter"},{"type":"retrieval"}],
        model="gpt-4-1106-preview"
    )

thread = client.beta.threads.create()      

prompt = st.chat_input("ask here...")
if prompt=="exit":
    st.stop()
if prompt:  
    with st.spinner("Thinking..."):  
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jatin. The user has a premium account."
        )

        while True:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run.status == "completed":
                break

        messages = client.beta.threads.messages.list(
        thread_id=thread.id
        )
    st.success("Done!")

    print(messages.data[0].content[0].text.value)
    response = messages.data[0].content[0].text.value

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    #bot.write(messages.data[0].content[0].text.value)