import openai
import tiktoken
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import wave
import cv2

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
from pathlib import Path
import tkinter as tk
import base64

from io import StringIO
from tkinter import messagebox

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
imageFlag = False
use_camera_flag = False
prompt = ""
createImageFlag = False
image_url = ""

# Define the basic parameters for the audio recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100              # Sampling rate
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 5       # Duration of recording
WAVE_OUTPUT_FILENAME = "recording.wav"  # Output filename

p = pyaudio.PyAudio()

def record_audio():
        print("Recording...")
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("Finished recording.")
        return True

def transcribe_audio():
    try:
        recording_path = "C:/Users/jatin/Documents/AI/base_1/recording.wav"
        transcript_text = ""
        
        audio_file = open(recording_path, "rb")
        transcript = st.session_state.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format='text'
        )
        
        # Extract the text from the response
        transcript_text = transcript['text'] if 'text' in transcript else 'Transcription failed'
        print(transcript)
        # Generate a new filename with datetime stamp and 'transcribed' suffix
        #new_filename = "C:/Users/jatin/Documents/AI/base_1/recording_{}_transcribed.wav".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        # Rename the file
        #os.rename(recording_path, new_filename)

        return transcript

    except Exception as e:
        # Log the exception details
        return "error"

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

def get_permission():
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Move the message box to the front
    root.attributes('-topmost', True)

    # Ask for permission
    permission = messagebox.askokcancel("Permission", "Do you want to take a picture?", parent=root)
    root.destroy()

    if permission:
        return True
    else:
        return False

def use_camera():
    # Ask for permission
    permission = get_permission()

    if permission:
        capture_image()
        return("permisssion granted. photo saved")
    else:
        return("permissiong not granted. photo not saved")
    
# Function for automatic brightness and contrast optimization using CLAHE
def auto_adjust_brightness_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return adjusted
    
def capture_image():
    global use_camera_flag
    global imageFlag

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # '0' is typically the default value for the laptop's built-in webcam

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    if ret:
        # Auto adjusting brightness and contrast
        frame = auto_adjust_brightness_contrast(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        cv2.waitKey()  # Wait for 2000 milliseconds (2 seconds)

        # Save the captured image to a file
        
        if not cv2.imwrite(r'C:\Users\jatin\Documents\AI\base_1\static\images\webcam_image.jpg', frame):
            raise RuntimeError("Unable to capture image")
        else:
            print("Image captured and saved successfully.")
            use_camera_flag =True
            imageFlag = True
    else:
        print("Failed to capture image")

    # Release the camera
    cap.release()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image():
    global use_camera_flag
    if use_camera_flag == False:
        capture_image()
    else:
        # Path to your image
        image_path = r"C:\Users\jatin\Documents\AI\base_1\static\images\webcam_image.jpg"

        # Getting the base64 string
        base64_image = encode_image(image_path)

        PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "Analyze this image"
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ],
        },
        ]   
        params = {
            "model": "gpt-4-vision-preview",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 500,
        }

        result = st.session_state.client.chat.completions.create(**params)
        vision_output = result.choices[0].message.content

    print("vision output: ", vision_output)
    return vision_output

def create_image():
    response = st.session_state.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
    
    image_url = response.data[0].url
    print("Image URL: ", image_url)
    if image_url != "":
        createImageFlag = True
    return image_url

def analyze_conversation(user_query_analysis, user_query, ai_response, thread_id, turn_count, start_time, end_time, conversation_history):
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = st.session_state.client.beta.assistants.create(
            name="Chat Analyzer",
            instructions = "You are a highly analytical and detail-oriented AI, designated as an AI Interaction Analyst. "
                "In this role, you will meticulously examine and interpret interactions between users and large "
                "language AI models. Your analysis will be instrumental in enhancing AI performance, enriching user "
                "experience, and influencing the strategic development of the AI system.\n\n"
                "Your input comprises messages exchanged between a user and an AI. From these messages, your analysis "
                "mus respond to the following tersely:\n\n"
                "1. Response to custom user question for analysis {user_query_analysis}\n"
                "2. **Top keywords**: Top keywords used by the user and the AI\n"
                "3. **Detailed Sentiment Analysis**: Identify the sentiment of the user, including emotional tones "
                "and intensity, and provide a normalized sentiment score.\n"
                "4. **Topic Analysis**: Catalogue topics discussed, their frequency, and context, highlighting the "
                "user's primary areas of interest.\n"
                "5. **Engagement & Curiosity Metrics**: Assess the level and nature of user engagement and curiosity "
                "throughout the conversation.\n"
                "6. **Behavioral Insights**: Deduce insights into user behavioral traits, such as speculative thinking, "
                "ethical considerations, and information processing style.\n"
                "7. **User Preferences & Predictions**: Offer predictive insights into the user's potential interests "
                "in products and content, based on the conversational topics.\n"
                "8. **AI Response Evaluation**: Evaluate the accuracy and relevance of AI responses in relation to "
                "subsequent user messages.\n"
                "9. **Ethical & Societal Concerns   **: Note any ethical and societal concerns raised by the user, "
                "particularly regarding technology and AI.\n"
                "10. **Content & Advertising Recommendations**: Provide suggestions for content themes and advertising "
                "strategies that align with the user's interests and discussions.\n"
                "11. **Conversational Dynamics**: Analyze the conversational style and dynamics, recommending "
                "optimizations for AI interactions to better suit the user's preferences and conversational style.\n"
                "12. **Identification of Misconceptions & Queries**: Identify areas where the user may have "
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
            print("Analyzing...")
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

# Initialize session state if not already done
if 'timestamps' not in st.session_state:
    st.session_state['timestamps'] = []

openai.api_key = os.getenv("OPEN_API_KEY")
if 'client' not in st.session_state:
    st.session_state.client = OpenAI()

st.title("OpenAI Bot")

with st.sidebar:
    st.title("Audio Input")
    input_audio_flag = st.toggle("Enable audio input", value=False)

    #st.title("Audio Output")
    #response_audio_flag = st.toggle("Enable audio output", value=True)
    response_audio_flag = True

    st.title("Upload files here")
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
        instructions="You are a greeter at the SWE 2023 conference. You have documents with information about the exhibitors."
                    "Use your knowledgre retrieval skills to answer questions about the exhibitors.",
        tools=[{"type": "code_interpreter"},{"type":"retrieval"},
                           {"type": "function",
             "function": {
                 "name": "use_camera",
                 "description": "Use the webcam camera to capture an image of the user or any object in the frame of the webcam.",
                 "parameters": {
                        "type": "object",
                        "properties": {
                                "user": {
                                "type": "string",
                                "description": "Name of the user"
                                },
                        },
                        "required": []
                }
            }
            },
            {"type": "function",
             "function": {
                 "name": "analyze_image",
                 "description": "Use this to analyze the image captured by the webcam and describe the image as reponse."
                 "It will automatically call the image capture function."
                 "Also use this also to see the user, surroudings, and objects in the scene.",
                 "parameters": {
                        "type": "object",
                        "properties": {
                                "user": {
                                "type": "string",
                                "description": "Name of the user"
                                },
                        },
                        "required": []
                }
            }
            },
            {"type": "function",
             "function": {
                 "name": "create_image",
                 "description": "Creates an image from the text input using Dall-e-3 image generation model of OpenAI",
                 "parameters": {
                        "type": "object",
                        "properties": {
                                "user": {
                                "type": "string",
                                "description": "Name of the user"
                                },
                        },
                        "required": []
                }
            }
            }],
        model="gpt-4-1106-preview"
    )

if 'thread' not in st.session_state:
    st.session_state.thread = st.session_state.client.beta.threads.create()      

if "prev_uploaded_file" not in st.session_state: 
    st.session_state.prev_uploaded_file = None

#prompt = ""
prompt_audio = ""
print("Input audio flag: ", input_audio_flag)
if input_audio_flag:
    with st.chat_message("assistant"):
        st.markdown("Recording for 5 seconds...")
    if record_audio()==True: 
        prompt_audio = transcribe_audio()
        if prompt_audio != "":
            prompt = prompt + "Audio prompt: " + prompt_audio + ". \n"
        print("Prompt audio: ", prompt_audio)

prompt_text = st.chat_input("Ask here...")

if prompt_text is not None:
    prompt = f"{prompt} \n Text Prompt: {prompt_text}"
print("Prompt: ", prompt)

if prompt != "": 
    start_time = datetime.now()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.status("Thinking...", expanded=False) as status:  
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            
            file_name = uploaded_file.name
            print(file_name)
            print(st.session_state.prev_uploaded_file)

            if st.session_state.prev_uploaded_file is not file_name:
                print("File changed")
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                st.success("PDF file saved successfully.")

                file_1 = st.session_state.client.files.create(
                    file=open(file_name, "rb"),
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
            status.update(label=run.status, state="running", expanded=False)
            if run.status == "completed":
                status.update(label=run.status, state="complete", expanded=False)
                break
            if run.status == "requires_action":
                msg=[]
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                print(tool_calls)
                for i in range(len(tool_calls)):
                    # if tool_calls[i].function.name == "get_location":
                    #     msg.append({
                    #         "tool_call_id": tool_calls[i].id,
                    #         "output": get_location()
                    #     })
                    # if tool_calls[i].function.name == "get_datetime":
                    #     msg.append({
                    #         "tool_call_id": tool_calls[i].id,
                    #         "output": get_datetime()
                    #     })
                    if tool_calls[i].function.name == "use_camera":
                        msg.append({
                            "tool_call_id": tool_calls[i].id,
                            "output": use_camera()
                        })
                    if tool_calls[i].function.name == "analyze_image":
                        msg.append({
                            "tool_call_id": tool_calls[i].id,
                            "output": analyze_image()
                        })
                    if tool_calls[i].function.name == "create_image":
                        msg.append({
                            "tool_call_id": tool_calls[i].id,
                            "output": create_image()
                        })
                print("tool output: ", msg)
                run = st.session_state.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=st.session_state.thread.id,
                    run_id=run.id,
                    tool_outputs=msg
                )

        messages = st.session_state.client.beta.threads.messages.list(
            thread_id=st.session_state.thread.id
        )

    print(messages.data[0].content[0].text.value)
    response = messages.data[0].content[0].text.value

    imageOpen = cv2.imread(r"C:\Users\jatin\Documents\AI\base_1\static\images\webcam_image.jpg")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
        if imageFlag: 
            st.image(imageOpen, channels="BGR", output_format="JPEG")
        if createImageFlag:
            st.image(image_url)

    input_audio_flag = False
    end_time = datetime.now()

    if response_audio_flag and response != "":
        text_input = response
        speech_file_path = Path(__file__).parent / "speech.mp3"

        response_audio = st.session_state.client.audio.speech.create(
            model="tts-1-1106",
            voice="onyx",
            input=text_input
        )
        response_audio.stream_to_file(speech_file_path)

        audio_file = open(speech_file_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')

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