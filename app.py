from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re
from common_skills import common_skills

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pipelines
ner_pipeline = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
feature_pipeline = pipeline("feature-extraction", model="ml6team/keyphrase-extraction-distilbert-inspec")

# Home route (optional)
@app.route('/')
def home():
    return "Resume NER API is running!"

# --- ROUTE 1: Basic NER Analysis ---
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    ner_results = ner_pipeline(text)
    return jsonify(ner_results)

# --- ROUTE 2: Structured Basic Resume Info ---
@app.route('/structured-analyze', methods=['POST'])
def structured_analyze():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    ner_results = ner_pipeline(text)

    structured = {
        "names": [],
        "organizations": [],
        "locations": [],
        "dates": [],
    }

    for entity in ner_results:
        group = entity.get("entity_group", "")
        word = entity.get("word", "")
        
        if group == "PER":
            structured["names"].append(word)
        elif group == "ORG":
            structured["organizations"].append(word)
        elif group == "LOC":
            structured["locations"].append(word)
        elif group == "DATE":
            structured["dates"].append(word)

    return jsonify(structured)

# --- ROUTE 3: Deep Structured Resume Analysis ---
@app.route('/deep-structured-analyze', methods=['POST'])
def deep_structured_analyze():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    ner_results = ner_pipeline(text)

    structured_resume = {
        "name": None,
        "emails": [],
        "phones": [],
        "skills": [],
        "experience": [],
        "education": [],
        "organizations": [],
        "dates": [],
        "locations": [],
    }

    # 1. Extract using NER
    for entity in ner_results:
        group = entity.get("entity_group", "")
        word = entity.get("word", "")
        
        if group == "PER" and not structured_resume["name"]:
            structured_resume["name"] = word
        elif group == "ORG":
            structured_resume["organizations"].append(word)
        elif group == "LOC":
            structured_resume["locations"].append(word)
        elif group == "MISC":
            structured_resume["education"].append(word)
        elif group == "DATE":
            structured_resume["dates"].append(word)

    # 2. Extract Email and Phone using regex
    email_regex = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    phone_regex = re.compile(r"(\+?\d{1,3})?[-.\s]?(\(?\d{3}\)?)[-.\s]?\d{3}[-.\s]?\d{4}")

    structured_resume["emails"] = email_regex.findall(text)
    structured_resume["phones"] = ["".join(phone) for phone in phone_regex.findall(text)]

    # 3. Extract Skills (basic filtering)
    keywords = []
    for word in text.split():
        if word.isalpha() and len(word) > 3:
            keywords.append(word.lower())

    # Example common skills list
    # common_skills = [
    #     "python", "javascript", "react", "nextjs", "nodejs", "aws", "docker",
    #     "machine learning", "sql", "mongodb", "html", "css", "flask", "django",
    #     "typescript", "tailwind", "pytorch", "tensorflow", "java", "c++", "c#"
    # ]
    

    structured_resume["skills"] = list(set(keywords).intersection(set(common_skills)))

    # 4. Experience (detect sentences mentioning years/months)
    experience_sentences = []
    for line in text.split('\n'):
        if re.search(r'\d+\s+(years|months)', line, re.IGNORECASE):
            experience_sentences.append(line.strip())

    structured_resume["experience"] = experience_sentences

    return jsonify(structured_resume)

# Main
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
