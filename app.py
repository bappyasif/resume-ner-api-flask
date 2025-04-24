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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)