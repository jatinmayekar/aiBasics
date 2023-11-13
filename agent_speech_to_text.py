from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os
from io import BytesIO
import openai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio part'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read file content into a BytesIO object
        print("processing file")
        file_stream = BytesIO()
        file.save(file_stream)
        file_stream.seek(0)  # Move to the beginning of the BytesIO object

        #file= open("C:/Users/jatin/Documents/AI/base_1/speech.mp3", "rb")
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=file
            )
            print(transcript)
            return jsonify({'transcription': transcript.text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
