import os
import fitz  # PyMuPDF
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def rank_resumes(resume_paths, job_description):
    jd_processed = preprocess_text(job_description)
    texts = [preprocess_text(extract_text_from_pdf(path)) for path in resume_paths]
    texts.append(jd_processed)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:]).flatten()

    data = [(os.path.basename(path), round(score, 3)) for path, score in zip(resume_paths, scores)]
    ranked = sorted(data, key=lambda x: x[1], reverse=True)
    return ranked