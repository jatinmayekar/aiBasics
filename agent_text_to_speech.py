from flask import Flask, render_template, request, jsonify, send_file
import openai
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Use the same OpenAI API key for both functionalities

@app.route('/')
def index():
    return render_template('index2.html')

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
