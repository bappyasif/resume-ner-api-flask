# app.py
import os, traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # opens CORS to all origins

# Lazy initialization
ner_pipeline = None
def init_pipeline():
    global ner_pipeline
    if ner_pipeline is None:
        ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            tokenizer="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )
    return ner_pipeline

@app.route("/", methods=["GET"])
def home():
    return "✅ Resume NER API is up!", 200

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    pipe = init_pipeline()
    try:
        raw = pipe(text)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Inference error", "details": str(e)}), 500

    # Convert to pure-Python
    out = []
    for ent in raw:
        out.append({
            "entity_group": ent.get("entity_group") or ent.get("entity") or ent.get("label"),
            "score": float(ent["score"]),
            "word": str(ent["word"]),
            "start": int(ent.get("start", 0)),
            "end":   int(ent.get("end", 0)),
        })

    return jsonify(out), 200

@app.route('/structured-analyze', methods=['POST'])
def structured_analyze_resume():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    ner_results = ner_pipeline(text)

    # Initialize empty fields
    structured_resume = {
        "name": None,
        "organizations": [],
        "locations": [],
        "dates": [],
        "misc": [],
    }

    for entity in ner_results:
        group = entity.get("entity_group", "")
        word = entity.get("word", "")

        if group == "PER" and not structured_resume["name"]:
            structured_resume["name"] = word  # First detected person as name
        elif group == "ORG":
            structured_resume["organizations"].append(word)
        elif group == "LOC":
            structured_resume["locations"].append(word)
        elif group == "MISC":
            structured_resume["misc"].append(word)
        elif group == "DATE":
            structured_resume["dates"].append(word)

    return jsonify(structured_resume)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)