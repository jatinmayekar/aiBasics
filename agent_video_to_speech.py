import cv2
import queue
import threading
import openai
import os
from pathlib import Path
import pygame
from datetime import datetime
import time
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# Define global variables
stop_threads = False
frame_queue = queue.Queue()
frame_interval = 3  # seconds
audio_playback_complete = threading.Event()

# Define the text-to-speech function
def text_to_speech(text_input):
    # Generate a unique filename for each speech file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    speech_file_path = Path(__file__).parent / f"speech_{timestamp}.mp3"
    print("Generating speech file:", speech_file_path)
    
    response = openai.audio.speech.create(
        model="tts-1-1106",
        voice="onyx",
        input=text_input
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

# Define the audio playback function
def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    audio_playback_complete.set()  # Signal that audio playback is complete

# Define the video capture thread
def video_capture_thread():
    global stop_threads
    vid = cv2.VideoCapture(0)

    while True:
        success, frame = vid.read()
        if not success:
            break

        frame_queue.put(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

    vid.release()
    cv2.destroyAllWindows()

# Define the frame processing thread
def frame_processing_thread():
    global stop_threads
    last_time = time.time()

    while not stop_threads:
        if not frame_queue.empty() and time.time() - last_time >= frame_interval:
            frame = frame_queue.get()
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")

            # Define the prompt for OpenAI API
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        "These are frames of a video. Create a terse voiceover text in Tom Cruise Voice. Only include the narration.",
                        {"image": base64_frame, "resize": 768}
                    ],
                },
            ]
            params = {
                "model": "gpt-4-vision-preview",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 500,
            }

            # Call OpenAI API and get the response
            result = client.chat.completions.create(**params)
            print(result.choices[0].message.content)

            # Convert the response text to speech
            unique_speech_file_path = text_to_speech(result.choices[0].message.content)
            print("Playing audio from:", unique_speech_file_path)

            # Play the audio and wait for completion
            play_audio(unique_speech_file_path)
            audio_playback_complete.wait()
            audio_playback_complete.clear()

            last_time = time.time()

# Create and start threads
capture_thread = threading.Thread(target=video_capture_thread)
processing_thread = threading.Thread(target=frame_processing_thread)

capture_thread.start()
processing_thread.start()

# Wait for threads to finish
capture_thread.join()
processing_thread.join()
