from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

import openai
import os
import json

load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual OpenAI API key
client = OpenAI()
debug=True

def get_email_address(email_address):
    return ("here is your latest email at " + email_address + ": " + "Hello, this is a test email.")

@app.route('/')
def index():
    return render_template('indexAll.html')

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

if __name__ == '__main__':
    app.run(debug=True)