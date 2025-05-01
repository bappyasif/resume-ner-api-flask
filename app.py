from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re

# Initialize Flask app
app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {
    "origins": ["http://localhost:3000", "[http://example.com](http://example.com)"],
    "methods": ["GET", "POST", "PUT", "DELETE"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type", "Authorization"],
    "max_age": 3600
}})

# ----- Utility: Extract experience section -----
def extract_experience_sections(text):
    patterns = [
        r"(work experience|professional experience|employment history)\s*[:\-]?\s*((.|\n)+?)(?=(skills|education|certifications|projects|$))",
        r"(experience)\s*[:\-]?\s*((.|\n)+?)(?=(skills|education|certifications|projects|$))",
    ]
    matches = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            extracted = match.group(2).strip()
            entries = re.split(r"\n\s*[\-\*\u2022]|\n\d{4}", extracted)
            matches.extend([entry.strip() for entry in entries if entry.strip()])
    return matches

# ----- Constants -----
common_skills = [
    # Programming languages
    "python", "javascript", "java", "c++", "c#", "swift", "ruby", "php", "go", "rust",
    
    # Front-end frameworks and libraries
    "react", "nextjs", "angular", "vue", "ember", "backbone", "jquery",
    
    # Back-end frameworks and libraries
    "nodejs", "express", "django", "flask", "ruby on rails", "laravel", "spring",
    
    # Databases
    "sql", "mongodb", "postgresql", "mysql", "redis", "cassandra", "couchbase",
    
    # Cloud platforms
    "aws", "azure", "google cloud", "heroku", "digitalocean",
    
    # Containerization and orchestration
    "docker", "kubernetes", "containerd", "rkt",
    
    # Machine learning and AI
    "machine learning", "deep learning", "pytorch", "tensorflow", "keras", "scikit-learn",
    
    # Web development tools and technologies
    "html", "css", "sass", "less", "stylus", "webpack", "gulp", "grunt",
    
    # Testing and debugging
    "jest", "mocha", "chai", "enzyme", "cypress", "selenium",
    
    # Version control
    "git", "svn", "mercurial",
    
    # Agile methodologies
    "scrum", "kanban", "lean",
    
    # Operating systems
    "linux", "windows", "macos",
    
    # Networking
    "http", "https", "tcp/ip", "udp", "dns", "dhcp",
    
    # Security
    "ssl", "tls", "oauth", "openid", "jwt", "security testing",
    
    # DevOps
    "continuous integration", "continuous deployment", "continuous monitoring",
    
    # Data science and analytics
    "data science", "data analytics", "data visualization", "data mining", "data warehousing",
    
    # Mobile app development
    "ios", "android", "react native", "flutter", "xamarin",
    
    # Desktop app development
    "electron", "nw.js", "qt", "wxwidgets",
    
    # Game development
    "unity", "unreal engine", "godot", "construct 3",
    
    # Virtual and augmented reality
    "vr", "ar", "mr", "xr",
    
    # Internet of things (IoT)
    "iot", "arduino", "raspberry pi", "esp32", "esp8266",
    
    # Blockchain
    "blockchain", "ethereum", "bitcoin", "smart contracts",
    
    # Cybersecurity
    "cybersecurity", "penetration testing", "vulnerability assessment", "incident response",
    
    # Artificial intelligence
    "ai", "natural language processing", "computer vision", "robotics",
    
    # Data engineering
    "data engineering", "data architecture", "data governance", "data quality",
    
    # Cloud engineering
    "cloud engineering", "cloud architecture", "cloud migration", "cloud security",
    
    # Full-stack development
    "full-stack development", "full-stack engineer", "full-stack developer",
    
    # UX/UI design
    "ux design", "ui design", "user experience", "user interface", "human-computer interaction",
    
    # Project management
    "project management", "project manager", "agile project management", "waterfall project management",
    
    # Technical writing
    "technical writing", "technical documentation", "api documentation", "user documentation",
    
    # Quality assurance
    "quality assurance", "qa engineer", "qa testing", "test automation",
    
    # DevOps engineering
    "devops engineering", "devops engineer", "devops automation", "devops monitoring",
    
    # Site reliability engineering
    "site reliability engineering", "sre engineer", "sre automation", "sre monitoring",
    
    # Technical leadership
    "technical leadership", "technical lead", "tech lead", "engineering manager",
    
    # Engineering management
    "engineering management", "engineering manager", "director of engineering", "vp of engineering",
    
    # Product management
    "product management", "product manager", "product owner", "product development",
    
    # Product design
    "product design", "product designer", "ux design", "ui design",
    
    # Business analysis
    "business analysis", "business analyst", "requirements gathering", "requirements analysis",
    
    # Data analysis
    "data analysis", "data analyst", "data scientist", "data engineer", "data warehouse",
    
    # Business intelligence
    "business intelligence", "bi developer", "bi analyst", "bi engineer",
    
    # Project management
    "project management", "project manager", "agile project management", "waterfall project management",
    
    # Technical writing
    "technical writing", "technical documentation", "api documentation", "user documentation",
    
    # Quality assurance
    "quality assurance", "qa engineer", "qa testing", "test automation",
    
    # DevOps engineering
    "devops engineering", "devops engineer", "devops automation", "devops monitoring",
    
    # Site reliability engineering
    "site reliability engineering", "sre engineer", "sre automation", "sre monitoring",
]

degree_keywords = [
    "bachelor", "master", "phd", "b.sc", "m.sc", "btech", "mtech", "mba", "msc", "bba", "bs", "ms"
]

# ----- Pipeline Init -----
resume_ner_pipeline = pipeline(
    'token-classification',
    model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
    aggregation_strategy='simple'
)

# Load model & tokenizer once
# summary_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
# summary_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Helper function to truncate based on tokenizer
# def safe_truncate_text(text, max_tokens=1024):
#     tokens = summary_tokenizer.encode(text, truncation=True, max_length=max_tokens, return_tensors="pt")
#     return summary_tokenizer.decode(tokens[0], skip_special_tokens=True)

# ----- Route -----
# @app.route('/resume-summary', methods=['POST', 'OPTIONS'])
# def resume_summary():
#     if request.method == 'OPTIONS':
#         return '', 200

#     data = request.get_json()
#     text = data.get('text', '')
#     if not text:
#         return jsonify({"error": "No text provided"}), 400

#     try:
#         # Optional: clean or format resume text
#         formatted_text = text.replace('\n', ' ').strip()
        
#         # Truncate properly to avoid token length errors
#         truncated_text = safe_truncate_text(formatted_text)

#         # Generate summary
#         summary = summary_pipeline(truncated_text)[0]["summary_text"]

#         return jsonify({"summary": summary})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# Lightweight summarization pipeline
summary_pipeline = pipeline("summarization", model="Falconsai/text_summarization")

@app.route('/resume-summary', methods=['POST', 'OPTIONS'])
def resume_summary():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Truncate text to avoid overflow (Falconsai handles longer than BART, but be cautious)
        max_input_length = 1024  # empirical limit for this model
        text = text[:max_input_length]

        summary = summary_pipeline(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/deep-structured-analyze', methods=['POST', 'OPTIONS'])
def deep_structured_analyze():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    ner_results = resume_ner_pipeline(text)

    structured_resume = {
        "name": None,
        "emails": [],
        "phones": [],
        "skills": [],
        "experience": [],
        "education": [],
        "organizations": [],
        "certifications": [],
        "dates": [],
        "locations": [],
    }

    # 1. NER entity extraction
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

    # 2. Email and Phone
    email_regex = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    phone_regex = re.compile(r"(\+?\d{1,3})?[-.\s]?(\(?\d{3}\)?)[-.\s]?\d{3}[-.\s]?\d{4}")

    structured_resume["emails"] = email_regex.findall(text)
    structured_resume["phones"] = ["".join(phone) for phone in phone_regex.findall(text)]

    # 3. Skills (keyword match)
    found_skills = []
    for skill in common_skills:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            found_skills.append(skill)
    structured_resume["skills"] = list(set(found_skills))

    # 4. Education (manual degree keyword scan)
    education_matches = []
    for line in text.split('\n'):
        if any(degree in line.lower() for degree in degree_keywords):
            education_matches.append(line.strip())
    structured_resume["education"].extend(list(set(education_matches)))

    # 5. Certifications
    cert_lines = []
    for line in text.split('\n'):
        if "certified" in line.lower() or "certificate" in line.lower() or "certification" in line.lower():
            cert_lines.append(line.strip())
    structured_resume["certifications"] = list(set(cert_lines))

    # 6. Experience: years/months line + section match
    exp_lines = [line.strip() for line in text.split('\n') if re.search(r'\d+\s+(years|months)', line, re.IGNORECASE)]
    exp_section = extract_experience_sections(text)
    structured_resume["experience"] = list(set(exp_lines + exp_section))

    return jsonify(structured_resume)

# ----- Optional Home Route -----
@app.route('/')
def home():
    return "Resume NER API is running!"

# ----- Run Server -----
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
