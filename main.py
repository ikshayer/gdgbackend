import os
import json
import random
import time
import base64
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel(model_name='gemini-2.0-flash')

@app.route("/get_description", methods=["POST"])
def sendInformation():

    body = request.get_json()
    
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    
    if not latitude or not longitude:
        return jsonify({"error": "No coordinates found"}), 400

    response = model.generate_content(f'Think that you are a geographical expert. I am going to give you coordinates, which are {"38.890452, -77.002375 (Dec Deg) 38° 53' 26'' N, 77° 0' 9'' W"}. You will give historic data about a geographical location and concise summary of it as a json object.')

    try:
        # Attempt to parse as JSON if possible
        data = json.loads(response.text)
        print(data)
    except json.JSONDecodeError:
        # If response isn't valid JSON, return plain text
        return jsonify({"raw_response": response.text}), 200

    return jsonify(data), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)