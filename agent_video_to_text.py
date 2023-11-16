from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
import openai
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

base64Frames = []
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    success, frame = vid.read() 
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    # Display the resulting frame 
    cv2.imshow('frame', frame)

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::60]),
            ],
        },
    ]   
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
