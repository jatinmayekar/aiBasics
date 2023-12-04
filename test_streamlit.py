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
if 'client' not in st.session_state:
    st.session_state.client = OpenAI()

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

if 'assistant' not in st.session_state:
    st.session_state.assistant = st.session_state.client.beta.assistants.create(
        name="Exhibitor Assistant",
        instructions="You are an greeter at the SWE 2023 conference. You have documents with information about the exhibitors."
                    "Use your knowledgre retrieval skills to answer questions about the exhibitors.",
        tools=[{"type": "code_interpreter"},{"type":"retrieval"}],
        model="gpt-4-1106-preview"
    )

if 'thread' not in st.session_state:
    st.session_state.thread = st.session_state.client.beta.threads.create()      

if "prev_uploaded_file" not in st.session_state: 
    st.session_state.prev_uploaded_file = None

prompt = st.chat_input("ask here...")
if prompt=="exit":
    st.stop()
if prompt: 
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.status("Thinking...", expanded=True) as status:  
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            
            file_name = uploaded_file.name
            #print(file_name)

            if st.session_state.prev_uploaded_file is not file_name:
                print("File changed")
                with open('swe_exhibitor_file.pdf', 'wb') as f:
                    f.write(bytes_data)
                st.success("PDF file saved successfully.")

                file_1 = st.session_state.client.files.create(
                    file=open("swe_exhibitor_file.pdf", "rb"),
                    purpose='assistants'
                )

                message = st.session_state.client.beta.threads.messages.create(
                    thread_id=st.session_state.thread.id,
                    role="user",
                    content=prompt,
                    file_ids=[file_1.id]
                )
            else:
                print("File not changed")
                message = st.session_state.client.beta.threads.messages.create(
                    thread_id=st.session_state.thread.id,
                    role="user",
                    content=prompt
                )
        elif uploaded_file is None:
            print("No file uploaded")
            message = st.session_state.client.beta.threads.messages.create(
                thread_id=st.session_state.thread.id,
                role="user",
                content=prompt
            )

        run = st.session_state.client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id,
            instructions="Please address the user as Jatin. The user has a premium account."
        )

        while True:
            time.sleep(2)
            run = st.session_state.client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=run.id
            )
            status.update(label=run.status, state="running", expanded=True)
            if run.status == "completed":
                status.update(label=run.status, state="complete", expanded=True)
                break

        messages = st.session_state.client.beta.threads.messages.list(
            thread_id=st.session_state.thread.id
        )
    #st.success("Done!")

    print(messages.data[0].content[0].text.value)
    response = messages.data[0].content[0].text.value

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    #bot.write(messages.data[0].content[0].text.value)
