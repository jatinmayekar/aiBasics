from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

import openai
import os

load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual OpenAI API key
client = OpenAI()
debug=False

@app.route('/')
def index():
    return render_template('indexAll.html')

@app.route('/text_to_text', methods=['POST'])
def submit():
    text_input = request.form['text_input']
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text_input}
        ]
    )    
    
    if debug: print(response)
    message_content = response.choices[0].message.content if response.choices else "No response"
    return jsonify(message_content)

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    text_input = request.form['text']
    speech_file_path = Path(__file__).parent / "speech.mp3"

    response = openai.audio.speech.create(
        model="tts-1-1106",
        voice="onyx",
        input=text_input
    )
    response.stream_to_file(speech_file_path)

    return send_file(speech_file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)