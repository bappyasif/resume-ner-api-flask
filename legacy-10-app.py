import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

import re
# from common_skills import common_skills

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

# Accessing the Hugging Face token from the environment variable
hf_token = os.getenv('HF_AUTH_TOKEN')

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

# Initialize the resume-ner pipeline (for /deep-structured-analyze)
resume_ner_pipeline = pipeline(
    'token-classification',
    # model='mrm8488/bert-mini-finetuned-conll2003-ner', // private!!
    model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
    aggregation_strategy='simple'
)

# Home route (optional)
@app.route('/')
def home():
    return "Resume NER API is running!"

# --- ROUTE 3: Deep Structured Resume Analysis ---
@app.route('/deep-structured-analyze', methods=['POST', 'OPTIONS'])
def deep_structured_analyze():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    ner_results = resume_ner_pipeline(text)
    # ner_results = feature_pipeline(text)

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
    

    structured_resume["skills"] = list(set(keywords).intersection(set(common_skills)))

    # 4. Experience (detect sentences mentioning years/months)
    experience_sentences = []
    # for line in text.split('\n'):
    #     if re.search(r'\d+\s+(years|months)', line, re.IGNORECASE):
    #         experience_sentences.append(line.strip())

    # structured_resume["experience"] = experience_sentences

     # 4.1 Experience (NER-based + regex section detection)
    experience_sentences = [line.strip() for line in text.split('\n') if re.search(r'\d+\s+(years|months)', line, re.IGNORECASE)]
    structured_resume["experience"] = list(set(experience_sentences + extract_experience_sections(text)))


    return jsonify(structured_resume)

# Main
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
