from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from datetime import datetime

import openai
import os
import json
import threading
import pyaudio
import wave
import time

load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual OpenAI API key
client = OpenAI()

def call_assistant():
    # Upload a file with an "assistants" purpose
    file1 = client.files.create(
        file=open("C:/Users/jatin/Downloads/swe_exhibitor_list.pdf", "rb"),
        purpose='assistants'
    )

    file2 = client.files.create(
        file=open("C:/Users/jatin/Downloads/SWE_Exhibitor_Information.pdf", "rb"),
        purpose='assistants'
    )

    assistant = client.beta.assistants.create(
        name="Exhibitor Assistant",
        instructions="You are an greeter at the SWE 2023 conference.",
        model="gpt-4-1106-preview",
        tools=[{"type": "code_interpreter"},{"type":"retrieval"}],
        file_ids=[file2.id]
    )

    thread = client.beta.threads.create()
    data = request.get_json()
    text_input = data['text_input']
    print(text_input)

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=text_input
    )
    #print(message)

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    runStatus = client.beta.threads.runs.retrieve(thread_id=thread.id,run_id=run.id)
    while run.status != "completed":
        time.sleep(1)
        runStatus = client.beta.threads.runs.retrieve(thread_id=thread.id,run_id=run.id)
        if runStatus.status == "completed":
            break

    responses = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    print(responses.data[0].content[0].text.value)

    if runStatus.status == 'completed':
        responses = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        print("reponses:\n") 
        print(responses)
        return jsonify(responses)
    else:
        return jsonify("No response")
    
if __name__ == '__main__':
    app.run(debug=True)