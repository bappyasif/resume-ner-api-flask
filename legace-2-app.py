from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load NER model pipeline
ner_pipeline = pipeline("token-classification", model="dslim/bert-base-NER", grouped_entities=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No input text provided"}), 400
    try:
        result = ner_pipeline(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Resume NER API is up!", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
