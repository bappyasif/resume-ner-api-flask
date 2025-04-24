# app.py
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

# ——— App & Logging Setup ———
app = Flask(__name__)
# Allow your front-end origins here (use "*" for all in dev)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "*"]}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ——— Initialize NER Pipeline ———
try:
    ner_pipeline = pipeline(
        task="ner",                         # alias for token-classification
        model="dslim/bert-base-NER",        # general-purpose NER model
        aggregation_strategy="simple"       # group tokens into entities
    )
    logger.info("✅ NER pipeline loaded successfully.")
except Exception as e:
    ner_pipeline = None
    logger.error("❌ Failed to load NER pipeline", exc_info=True)

# ——— Health Check ———
@app.route("/", methods=["GET"])
def home():
    return "✅ Resume NER API is up and running!", 200

# ——— NER Endpoint ———
@app.route("/analyze", methods=["POST"])
def analyze():
    if ner_pipeline is None:
        return jsonify({"error": "NER pipeline not initialized"}), 500

    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No input text provided"}), 400

    try:
        result = ner_pipeline(text)
        return jsonify(result)
    except Exception as e:
        # Log full stacktrace to Render logs
        logger.error("Error during NER inference", exc_info=True)
        # Return a user-friendly error plus details for debugging
        return (
            jsonify({
                "error": "Internal error during NER inference",
                "details": str(e)
            }),
            500,
        )

# ——— Entry Point ———
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Bind to 0.0.0.0 so Render can detect the port
    app.run(host="0.0.0.0", port=port)
# ——— END OF FILE ———