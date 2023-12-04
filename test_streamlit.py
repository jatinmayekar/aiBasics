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
import logging
from datetime import datetime

from io import StringIO

import load_dotenv
from load_dotenv import load_dotenv

load_dotenv()

# Initialize logging
logging.basicConfig(filename='chatgpt_analyzer.log', level=logging.INFO)

openai.api_key = os.getenv("OPEN_API_KEY")
if 'client' not in st.session_state:
    st.session_state.client = OpenAI()

st.title("OpenAI Bot")

# Function to log conversation and other details
def log_conversation_details(user_query, ai_response, thread_id, turn_count, start_time, end_time):
    # Basic Interaction Data
    logging.info(f"Timestamp: {datetime.now()}, User Request: {user_query}, AI Response: {ai_response}")

    # Conversation Context
    logging.info(f"Thread ID: {thread_id}, Turn Count: {turn_count}")

    # User Interaction Patterns
    duration = (end_time - start_time).total_seconds()
    logging.info(f"Response Time: {duration} seconds")

# Placeholder functions for sentiment and topic analysis
def analyze_conversation(conversation_history):
    # Implement sentiment analysis
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = st.session_state.client.beta.assistants.create(
            name="Chat Analyzer",
            instructions="You are an highly analytical and detail-oriented AI acting as AI Interaction Analyst." 
            "In this role, you will be responsible for meticulously examining and interpreting the interactions between users and large language AI models." 
            "Your insights will be pivotal in enhancing this AI's performance, improving user experience, and guiding the strategic direction of this AI's development."
            "Your input will be given the messages between a user and an AI."
            "Your output should be to sentiment of the user and a normalized sentiment score, topics discussed, frequency of discussion of these topics,"
            "accuracy of AI responses based on subsequent user messages, issues/error/feedbacks/pain points from the user based on the user messages, "
            "final direction to optimize the AI to suit the user",
            tools=[{"type": "code_interpreter"},{"type":"retrieval"}],
            model="gpt-4-1106-preview"
        )

    if 'analyzer_thread' not in st.session_state:
        st.session_state.analyzer_thread = st.session_state.client.beta.threads.create()   

    analyzer_message = st.session_state.client.beta.threads.messages.create(
        thread_id=st.session_state.analyzer_thread.id,
        role="user",
        content=conversation_history
    )

    analyzer_run = st.session_state.client.beta.threads.runs.create(
        thread_id=st.session_state.analyzer_thread.id,
        assistant_id=st.session_state.analyzer.id,
    )

    while True:
        print("analyzing")
        time.sleep(2)
        analyzer_run_retrieve = st.session_state.client.beta.threads.runs.retrieve(
            thread_id=st.session_state.analyzer_thread.id,
            run_id=analyzer_run.id
        )
        if analyzer_run_retrieve.status == "completed":
            break

    analyzer_response = st.session_state.client.beta.threads.messages.list(
        thread_id=st.session_state.analyzer_thread.id
    )

    analyzer_response_value = analyzer_response.data[0].content[0].text.value

    return analyzer_response_value

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
    start_time = datetime.now()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.status("Thinking...", expanded=True) as status:  
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            
            file_name = uploaded_file.name
            print(file_name)
            print(st.session_state.prev_uploaded_file)

            if st.session_state.prev_uploaded_file is not file_name:
                print("File changed")
                with open('swe_exhibitor_list.pdf', 'wb') as f:
                    f.write(bytes_data)
                st.success("PDF file saved successfully.")

                file_1 = st.session_state.client.files.create(
                    file=open("swe_exhibitor_list.pdf", "rb"),
                    purpose='assistants'
                )

                st.session_state.prev_uploaded_file = file_name

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

    end_time = datetime.now()

    log_conversation_details(user_query=prompt, ai_response=response, 
                             thread_id=st.session_state.thread.id, turn_count=len(st.session_state.messages),
                             start_time=start_time, end_time=end_time)

    print("Conversation history: ", json.dumps(st.session_state.messages))

    # Analyze sentiment and topics
    analyzer_response_log = analyze_conversation(json.dumps(st.session_state.messages))
    print("Analyzer reponse log: ", analyzer_response_log)

    # Additional logging for analysis
    logging.info(f"Analyzer: {analyzer_response_log}")
