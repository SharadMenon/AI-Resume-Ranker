import os
import re
import pymupdf as fitz
from sentence_transformers import SentenceTransformer, util

# Load SentenceTransformer model once on startup
model = SentenceTransformer('all-MiniLM-L6-v2')

# Skill taxonomy grouped by category with aliases
SKILL_CATEGORIES = {
    'Core Programming Languages': {
        'Python': ['python', 'py'],
        'Java': ['java'],
        'C++': ['c++', 'cpp'],
        'C# / .NET': ['c#', '.net', 'dotnet'],
        'Go / Golang': ['go', 'golang'],
        'JavaScript / TypeScript': ['javascript', 'js', 'typescript', 'ts'],
    },
    'Web & Fullstack Frameworks': {
        'React': ['react', 'reactjs', 'react.js'],
        'Angular': ['angular', 'angularjs'],
        'Node.js / Express': ['node.js', 'nodejs', 'node', 'express', 'express.js'],
        'Flask / FastAPI': ['flask', 'fastapi'],
        'Django': ['django'],
        'HTML / CSS': ['html', 'html5', 'css', 'css3'],
    },
    'Databases & Backend': {
        'SQL / Relational DBs': ['sql', 'postgresql', 'postgres', 'mysql', 'sqlite'],
        'NoSQL / MongoDB': ['mongodb', 'nosql', 'mongoose'],
        'APIs / Microservices': ['rest apis', 'rest api', 'api', 'apis', 'microservices', 'socket.io', 'jwt'],
    },
    'AI, LLMs & GenAI': {
        'Generative AI / LLMs': ['generative ai', 'genai', 'llm', 'llms', 'large language models', 'watsonx', 'watsonx.ai', 'ollama', 'transformers'],
        'RAG': ['rag', 'retrieval augmented generation'],
        'Vector DBs & Embeddings': ['vector databases', 'vector database', 'chromadb', 'faiss', 'pinecone', 'embeddings', 'sentence transformers', 'semantic search'],
        'AI Agents & Frameworks': ['ai agents', 'ai agent', 'agentic', 'langgraph', 'crewai', 'autogen', 'langchain'],
        'AI Coding Assistants': ['claude code', 'github copilot', 'copilot', 'cursor'],
        'NLP & Machine Learning': ['nlp', 'natural language processing', 'spacy', 'nltk', 'scikit-learn', 'tf-idf', 'tokenization', 'machine learning'],
        'Computer Vision / Media': ['opencv', 'whisper', 'ffmpeg', 'vision transformers'],
    },
    'Cloud, DevOps & Tools': {
        'Cloud Platforms': ['aws', 'azure', 'gcp', 'google cloud'],
        'DevOps / Containers / CI/CD': ['docker', 'kubernetes', 'ci/cd', 'cicd'],
        'Git / GitHub': ['git', 'github', 'gitlab'],
    },
    'Computer Science Foundations': {
        'CS Fundamentals / OOP': ['data structures', 'algorithms', 'oop', 'object oriented programming', 'computer science'],
    }
}

# Flat list of all skill names for quick canonical lookup
ALL_SKILLS = {}
for category, skills in SKILL_CATEGORIES.items():
    for skill_name, aliases in skills.items():
        ALL_SKILLS[skill_name] = (category, aliases)

def extract_text_from_pdf(path):
    """Extract full text from a PDF file using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text() + "\n"
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return text

def extract_skills_from_text(text):
    """Extract canonical skills from text based on alias and pattern matching."""
    text_lower = ' ' + text.lower() + ' '
    found_skills = set()
    
    for skill_name, (category, aliases) in ALL_SKILLS.items():
        for alias in aliases:
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(alias) + r'(?![a-zA-Z0-9_])'
            if re.search(pattern, text_lower):
                found_skills.add(skill_name)
                break
    return found_skills

def calculate_category_fulfillment(jd_skills, resume_skills):
    """
    Calculates skill match considering that many JD requirements are 'OR' options
    (e.g., 'Python OR Java OR C++', 'React OR Angular').
    """
    if not jd_skills:
        return 1.0, [], []
        
    matched = set()
    missing = set()
    
    # Categorize required skills
    jd_categories = {}
    for skill in jd_skills:
        cat = ALL_SKILLS[skill][0]
        jd_categories.setdefault(cat, []).append(skill)

    total_categories = len(jd_categories)
    fulfilled_categories_score = 0.0

    for cat, skills in jd_categories.items():
        cat_matched = [s for s in skills if s in resume_skills]
        cat_missing = [s for s in skills if s not in resume_skills]
        
        matched.update(cat_matched)
        
        # In Core Programming and Web Frameworks, having 1+ option fulfills the requirement
        if cat in ['Core Programming Languages', 'Web & Fullstack Frameworks']:
            if cat_matched:
                fulfilled_categories_score += 1.0
            else:
                missing.update(cat_missing)
        else:
            # For AI/GenAI and specialized categories, score by proportion matched
            proportion = len(cat_matched) / len(skills)
            fulfilled_categories_score += proportion
            missing.update(cat_missing)

    category_score = (fulfilled_categories_score / total_categories) if total_categories > 0 else 0.0
    return category_score, sorted(list(matched)), sorted(list(missing))

def rank_resumes(resume_paths, job_description):
    """
    Ranks resumes against a job description using:
    1. Sentence-Transformers semantic embedding similarity
    2. Intelligent category & skill fulfillment
    """
    if not resume_paths or not job_description.strip():
        return []

    jd_text = job_description.strip()
    jd_skills = extract_skills_from_text(jd_text)

    # Encode JD using SentenceTransformer
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)

    results = []

    for path in resume_paths:
        resume_text = extract_text_from_pdf(path)
        if not resume_text.strip():
            continue
            
        resume_skills = extract_skills_from_text(resume_text)
        
        # 1. Semantic Cosine Similarity
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        raw_sim = util.cos_sim(resume_embedding, jd_embedding).item()
        
        # Scale semantic cosine similarity (0.20 -> 0%, 0.60+ -> 95-100%)
        semantic_percentage = max(0.0, min(100.0, ((raw_sim - 0.20) / 0.40) * 100))
        
        # 2. Skill & Category Fulfillment Score
        cat_score, matched_skills, missing_skills = calculate_category_fulfillment(jd_skills, resume_skills)
        skill_percentage = cat_score * 100

        # 3. Hybrid Final Score (50% Semantic understanding + 50% Skill/Category Fulfillment)
        final_score = round(0.50 * semantic_percentage + 0.50 * skill_percentage, 1)
        final_score = max(5.0, min(98.5, final_score))

        # Match category badge
        if final_score >= 75:
            match_level = 'Strong Match'
            badge_color = '#10b981' # emerald
        elif final_score >= 50:
            match_level = 'Moderate Match'
            badge_color = '#3b82f6' # blue
        elif final_score >= 35:
            match_level = 'Partial Match'
            badge_color = '#f59e0b' # amber
        else:
            match_level = 'Low Match'
            badge_color = '#ef4444' # red

        results.append({
            'filename': os.path.basename(path),
            'score': final_score,
            'semantic_score': round(semantic_percentage, 1),
            'skill_score': round(skill_percentage, 1),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'match_level': match_level,
            'badge_color': badge_color
        })

    # Sort descending by final score
    ranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return ranked_results