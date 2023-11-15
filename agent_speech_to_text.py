from flask import Flask, request, jsonify, render_template
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

@app.route('/')
def index():
    return render_template('index3.html')  # Replace with your HTML filename

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio part'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the audio file
    filepath = os.path.join('uploads', audio_file.filename)
    audio_file.save(filepath)

    try:
        audio_file= open("C:/Users/jatin/Documents/AI/base_1/uploads/Recording.mp3", "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        return jsonify({'transcription': transcript.text})
    except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    #return jsonify({'message': 'File uploaded successfully', 'filepath': filepath})

if __name__ == '__main__':
    app.run(debug=True)
