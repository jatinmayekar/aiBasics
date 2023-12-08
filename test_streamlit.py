import openai
import tiktoken
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from io import StringIO

import load_dotenv
from load_dotenv import load_dotenv

load_dotenv()

# Initialize logging
logging.basicConfig(filename='chatgpt_analyzer.log', level=logging.INFO)

# Suppress info logging from OpenAI API only warnings and errors will still be logged
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

ai_model = "gpt-4-1106-preview"
user_id = 46
user_query=""
ai_response="" 
thread_id=0 
turn_count=0
start_time = datetime.now()
end_time=datetime.now()
conversation_history=""
analyzer_response_value_display=""
#encoding = tiktoken.get_encoding("cl100k_base")
user_token = 0
ai_token = 0
total_tokens = 0

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# Initialize session state if not already done
if 'timestamps' not in st.session_state:
    st.session_state['timestamps'] = []

openai.api_key = os.getenv("OPEN_API_KEY")
if 'client' not in st.session_state:
    st.session_state.client = OpenAI()

st.title("OpenAI Bot")

# Function to log conversation and other details
def analyze_conversation(user_query_analysis, user_query, ai_response, thread_id, turn_count, start_time, end_time, conversation_history):
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = st.session_state.client.beta.assistants.create(
            name="Chat Analyzer",
            instructions = "You are a highly analytical and detail-oriented AI, designated as an AI Interaction Analyst. "
                "In this role, you will meticulously examine and interpret interactions between users and large "
                "language AI models. Your analysis will be instrumental in enhancing AI performance, enriching user "
                "experience, and influencing the strategic development of the AI system.\n\n"
                "Your input comprises messages exchanged between a user and an AI. From these messages, your analysis "
                "should produce:\n\n"
                "1. Response to custom user question for analysis {user_query_analysis}\n"
                "1. **Top keywords**: Top keywords used by the user and the AI\n"
                "1. **Detailed Sentiment Analysis**: Identify the sentiment of the user, including emotional tones "
                "and intensity, and provide a normalized sentiment score.\n"
                "2. **Topic Analysis**: Catalogue topics discussed, their frequency, and context, highlighting the "
                "user's primary areas of interest.\n"
                "3. **Engagement & Curiosity Metrics**: Assess the level and nature of user engagement and curiosity "
                "throughout the conversation.\n"
                "4. **Behavioral Insights**: Deduce insights into user behavioral traits, such as speculative thinking, "
                "ethical considerations, and information processing style.\n"
                "5. **User Preferences & Predictions**: Offer predictive insights into the user's potential interests "
                "in products and content, based on the conversational topics.\n"
                "6. **AI Response Evaluation**: Evaluate the accuracy and relevance of AI responses in relation to "
                "subsequent user messages.\n"
                "7. **Ethical & Societal Concerns**: Note any ethical and societal concerns raised by the user, "
                "particularly regarding technology and AI.\n"
                "8. **Content & Advertising Recommendations**: Provide suggestions for content themes and advertising "
                "strategies that align with the user's interests and discussions.\n"
                "9. **Conversational Dynamics**: Analyze the conversational style and dynamics, recommending "
                "optimizations for AI interactions to better suit the user's preferences and conversational style.\n"
                "10. **Identification of Misconceptions & Queries**: Identify areas where the user may have "
                "misconceptions or unique viewpoints, suggesting potential areas for clarification or further information.\n\n"
                "Your objective is to extend beyond enhancing user experience. Aim to unearth deep insights into the "
                "user's preferences, interests, and engagement patterns. This information is invaluable for tailoring "
                "personalized content and developing effective advertising strategies that resonate with the user.",
            tools=[{"type": "code_interpreter"},{"type":"retrieval"}],
            model=ai_model
        )

    if 'analyzer_thread' not in st.session_state:
        st.session_state.analyzer_thread = st.session_state.client.beta.threads.create()   

    with st.status("Analyzing...", expanded=False) as status:
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
            status.update(label=analyzer_run_retrieve.status, state="running", expanded=False)
            if analyzer_run_retrieve.status == "completed":
                status.update(label=analyzer_run_retrieve.status, state="complete", expanded=False)
                break

        analyzer_response = st.session_state.client.beta.threads.messages.list(
            thread_id=st.session_state.analyzer_thread.id
        )

        analyzer_response_value = analyzer_response.data[0].content[0].text.value
        print("Analyzer response: ", analyzer_response)
    
    # user_token_length = num_tokens_from_string(user_query, encoding)
    # ai_token_length = num_tokens_from_string(ai_response, encoding)
    # total_tokens = num_tokens_from_messages(st.session_state.messages, model=ai_model)

    user_token_length = 0
    ai_token_length = 0
    total_tokens = 0

    log_entry = {
        "timestamp": str(datetime.now()),
        "thread_id": thread_id,
        "turn_count": turn_count,
        "response_time_seconds": (end_time - start_time).total_seconds(),
        "user_request": user_query,
        "ai_response": ai_response,
        "user_token_length": user_token_length,
        "ai_token_length": ai_token_length,
        "total_tokens": total_tokens,
        "analyzer_response": analyzer_response_value
    }

    logging.info(json.dumps(log_entry))
    return analyzer_response_value

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")   

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

    print(messages.data[0].content[0].text.value)
    response = messages.data[0].content[0].text.value

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    end_time = datetime.now()

    #print("Conversation history: ", json.dumps(st.session_state.messages))

    user_query=prompt
    ai_response=response
    thread_id=st.session_state.thread.id
    turn_count=len(st.session_state.messages)
    start_time = start_time
    end_time=end_time
    conversation_history=json.dumps(st.session_state.messages)

    # # Analyze sentiment and topics
    # analyzer_response_value_display=analyze_conversation(user_query_analysis, user_query=prompt, ai_response=response, 
    #                          thread_id=st.session_state.thread.id, turn_count=len(st.session_state.messages),
    #                          start_time=start_time, end_time=end_time, conversation_history=json.dumps(st.session_state.messages))

with st.sidebar:
    st.title("Analyzer")
    user_query_analysis = st.text_input('Add your custom user question for analysis here:')
    print("User query analysis: ", user_query_analysis)
    if st.button("Analyze"):
        # Analyze sentiment and topics
        analyzer_response_value_display=analyze_conversation(user_query_analysis=user_query_analysis, user_query=prompt, ai_response=ai_response, 
                                thread_id=st.session_state.thread.id, turn_count=len(st.session_state.messages),
                                start_time=start_time, end_time=end_time, conversation_history=json.dumps(st.session_state.messages))
        st.write("User ID ", user_id)
        st.write("Model: gpt-4-1106-preview")
        st.write("Timestamp ", start_time)

        current_time = pd.Timestamp.now()  # or use any other method to get the current time
        st.session_state['timestamps'].append(current_time)

        # Convert to DataFrame
        df = pd.DataFrame({'Timestamp': st.session_state['timestamps']})

        # Plotting (if there are timestamps)
        if not df.empty:
            # Extracting hour of the day for daily analysis
            df['Hour'] = df['Timestamp'].dt.hour

            # Plotting Hourly Distribution
            plt.figure(figsize=(10, 4))
            sns.histplot(df['Hour'], bins=24, kde=False)
            plt.title('Hourly Interaction Frequency')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Number of Interactions')
            plt.xticks(range(0, 24))
            plt.grid(True)

            # Display the plot in Streamlit
            st.pyplot(plt)

        response_time_seconds = (end_time - start_time).total_seconds()
        
        st.write("Thread ID: ", thread_id)
        st.write("Count: ", turn_count)
        st.write("response_time_seconds ",str(response_time_seconds))
        st.write("conversation_history ", conversation_history)
        st.write("analyzer_response_value ", analyzer_response_value_display)
    else:
        st.write("Click on Analyze to get the analysis")
