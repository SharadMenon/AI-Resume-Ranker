# AI-Powered Resume Ranker

An intelligent web application that ranks resumes based on their relevance to a provided job description using Natural Language Processing (NLP) techniques.

---

## 🔍 Features

* Upload multiple resumes (PDF)
* Paste job description in text area
* Extracts and preprocesses text using SpaCy
* Uses TF-IDF and cosine similarity to rank resumes
* Displays ranked candidates with similarity scores
* Simple, styled web UI with Flask + HTML/CSS

---

## 🛠 Technologies Used

* **Python**
* **Flask** – for web app and routing
* **SpaCy** – for NLP preprocessing
* **scikit-learn** – for TF-IDF and cosine similarity
* **PyMuPDF (fitz)** – for PDF text extraction
* **HTML/CSS + JS** – for frontend

---

## 📂 Project Structure

```
resume_ranker/
├── app.py                 # Flask app logic
├── utils.py               # Core functions (NLP, scoring)
├── templates/
│   └── index.html         # Webpage template
├── static/
│   └── style.css          # Styling
├── uploads/               # Uploaded resumes (temp storage)
├── job_description.txt    # (Optional) JD test input
└── README.md
```

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/resume-ranker.git
cd resume-ranker
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the Flask App

```bash
python app.py
```

### 4. Open in Browser

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---


## ✅ Status

* [x] NLP pipeline working
* [x] Resume scoring and ranking
* [x] Functional web interface
* [x] Responsive styling and JS interactivity

---

## 🧠 Future Enhancements

* Keyword highlighting
* Skill-based tagging
* Export to CSV
* Integration with email or LinkedIn
* Deploy on Render or Railway

---
