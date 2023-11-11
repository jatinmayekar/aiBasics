from flask import Flask, render_template, request, jsonify
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual OpenAI API key
client = OpenAI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    text_input = request.form['text_input']
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text_input}
        ]
    )
    # response = client.completions.create(
    # model="gpt-3.5-turbo-instruct",
    # prompt=text_input
    # )
    
    print(response)

    # message_content = response.choices[0].text.strip() if response.choices else "No response"
    message_content = response.choices[0].message.content if response.choices else "No response"
    
    return jsonify(message_content)

if __name__ == '__main__':
    app.run(debug=True)
