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
import geocoder
from datetime import datetime
import numpy as np

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# Define global variables
stop_threads = False
frame_queue = queue.Queue()
frame_interval = 1/30  # seconds
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

# Function for automatic brightness and contrast optimization using CLAHE
def auto_adjust_brightness_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return adjusted

def location():
    g = geocoder.ip('me')
    city = g.city
    country = g.country
    zipcode = g.postal
    location_details = "You are in " + city + ", " + country + ". Your zipcode is " + zipcode + "."
    return location_details

def datetimedetails():
    datetimeobject = datetime.now()
    date = datetimeobject.strftime("%d-%m-%Y")
    time = datetimeobject.strftime("%H:%M:%S")
    date_time = "Today's date is " + date + " and the time is " + time + "."
    return date_time

def is_frame_different(frame1, frame2, threshold=10):
    # Convert frames to grayscale for comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Check if the average difference across the frame is above the threshold
    return np.mean(diff) > threshold

# Define the video capture thread
def video_capture_thread():
    global stop_threads
    vid = cv2.VideoCapture(0)

    while True:
        success, frame = vid.read()
        if not success:
            break

        # Auto adjusting brightness and contrast
        frame = auto_adjust_brightness_contrast(frame)

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
    last_processed_frame = None

    while not stop_threads and time.time() - last_time >= frame_interval:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Only process the frame if it's significantly different from the last one
            if last_processed_frame is None or is_frame_different(last_processed_frame, frame):
                _, buffer = cv2.imencode(".jpg", frame)
                base64_frame = base64.b64encode(buffer).decode("utf-8")

                # Define the prompt for OpenAI API
                PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": [
                            "These are frames of a video. Create a voice over to describe the objects, human beings, events, and actions in the frame. Only include the narration.",
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
                #print(result.choices[0].message.content)

                # Combine the response text with location and date-time details
                finalstring = result.choices[0].message.content + " " + location() + " " + datetimedetails()
                print(finalstring)

                # Convert the response text to speech
                unique_speech_file_path = text_to_speech(finalstring)
                print("Playing audio from:", unique_speech_file_path)

                # Play the audio and wait for completion
                play_audio(unique_speech_file_path)
                audio_playback_complete.wait()
                audio_playback_complete.clear()

                # Update the last processed frame
                last_processed_frame = frame.copy()
                
            last_time = time.time()

# Create and start threads
capture_thread = threading.Thread(target=video_capture_thread)
processing_thread = threading.Thread(target=frame_processing_thread)

capture_thread.start()
processing_thread.start()

# Wait for threads to finish
capture_thread.join()
processing_thread.join()
