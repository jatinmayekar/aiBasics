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

# Define the basic parameters for the audio recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100              # Sampling rate
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 5       # Duration of recording
WAVE_OUTPUT_FILENAME = "recording.wav"  # Output filename

load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual OpenAI API key
client = OpenAI()
debug=True
# Initialize PyAudio
p = pyaudio.PyAudio()
# Global variable to indicate when recording is done
is_recording_complete = False

def get_audio_device_list():
    # Print the list of available devices and their info
    print("Available devices:\n")
    for i in range(p.get_device_count()):
        print(f"Device {i}: {p.get_device_info_by_index(i).get('name')}")

@app.route('/start_recording', methods=['GET'])
def start_recording():
    global is_recording_complete
    is_recording_complete = False
    transcription=""
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
        is_recording_complete = True
        print("Finished recording.")

    
    def transcribe_audio():
        try:
            recording_path = "C:/Users/jatin/Documents/AI/base_1/recording.wav"
            transcript_text = ""
            
            audio_file = open(recording_path, "rb")
            transcript = client.audio.transcriptions.create(
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
            app.logger.error('Exception: %s', str(e))
            # Return a JSON response with the error message and a 500 status code
            return jsonify({'error': str(e)}), 500

    # Start recording and wait for it to finish
    record_audio()

    print("here out: " + str(is_recording_complete))
    # Get the transcription
    if is_recording_complete==True: 
        print("here")
    transcription = transcribe_audio()
    return jsonify({'message': transcription})

def transcribe_audio():
    try:
        recording_path = "C:/Users/jatin/Documents/AI/base_1/recording.wav"
        transcript_text = ""
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        print(file.content_type)

        if file:
            with open(recording_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            transcript_text = transcript.text

            # Generate a new filename with datetime stamp and 'transcribed' suffix
            new_filename = "C:/Users/jatin/Documents/AI/base_1/recording_{}_transcribed.wav".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
            # Rename the file
            os.rename(recording_path, new_filename)

            return transcript_text

    except Exception as e:
        # Log the exception details
        app.logger.error('Exception: %s', str(e))
        # Return a JSON response with the error message and a 500 status code
        return jsonify({'error': str(e)}), 500

def get_email_address(email_address):
    return ("here is your latest email at " + email_address + ": " + "Hello, this is a test email.")

@app.route('/')
def index():
    return render_template('indexAll.html')

@app.route('/start_recording', methods=['GET'])
def handle_start_recording():
    message = start_recording()
    return jsonify({'message': message})

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text_input = data['text']
    speech_file_path = Path(__file__).parent / "speech.mp3"

    response = openai.audio.speech.create(
        model="tts-1-1106",
        voice="onyx",
        input=text_input
    )
    response.stream_to_file(speech_file_path)

    if debug: print("Audio file path: " + str(speech_file_path))

    return send_file(speech_file_path, as_attachment=True)

@app.route('/text_to_text_function', methods=['POST'])
def submit():
    data = request.get_json()
    text_input = data['text_input']
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": text_input}
    ]
    tools = [
    {
        "type": "function",
        "function": {
            "name": "get_email_address",
            "description": "Retrieve the most recent email from a specified email address. This function checks the inbox of the provided email address and returns the latest email, ensuring the user stays updated on their most recent correspondence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_address": {
                        "type": "string",
                        "description": "The email address whose inbox will be checked for the latest email.",
                    },
                },
                "required": ["email_address"],
            },
        },
    }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message_content = response.choices[0].message.content if response.choices else "No response"
    if debug: print("Text to text & function first output:" + str(message_content))

    response_msg = response.choices[0].message
    tool_calls = response_msg.tool_calls
    second_response = None

     # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_email_address": get_email_address,
        }  # only one function in this example, but you can have multiple
        messages.append(response_msg)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                email_address=function_args.get("email_address"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  # get a new response from the model where it can see the function responses

    if second_response == None:
        second_response = response
    
    if debug: print("Text to text & function second output:" + str(second_response.choices[0].message.content))

    return jsonify(second_response.choices[0].message.content if second_response.choices else "No response")

@app.route('/linkAssistant', methods=['POST'])
def callAssistant():
    # Upload a file with an "assistants" purpose
    file1 = client.files.create(
        file=open("C:/Users/jatin/Downloads/swe.txt", "rb"),
        purpose='assistants'
    )

    assistant = client.beta.assistants.create(
        name="Exhibitor Assistant",
        instructions="You are an greeter at the SWE 2023 conference. You have documents with information about the exhibitors."
              "Use your knowledgre retrieval skills to answer questions about the exhibitors. ",
        model="gpt-4-1106-preview",
        tools=[{"type": "code_interpreter"},{"type":"retrieval"}],
        file_ids=[file1.id]
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

    return jsonify(responses.data[0].content[0].text.value)
    
if __name__ == '__main__':
    app.run(debug=True)