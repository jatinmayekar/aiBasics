from flask import Flask, request, jsonify
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
from flask_cors import CORS
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        recording_path = "C:/Users/jatin/Downloads/recording.mp3"
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
            new_filename = "C:/Users/jatin/Downloads/recording_{}_transcribed.mp3".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
            # Rename the file
            os.rename(recording_path, new_filename)

            return jsonify({'transcription': transcript_text})

    except Exception as e:
        # Log the exception details
        app.logger.error('Exception: %s', str(e))
        # Return a JSON response with the error message and a 500 status code
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=8000)
