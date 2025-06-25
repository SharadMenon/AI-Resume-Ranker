from flask import Flask, render_template, request
import os
from utils import rank_resumes

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    ranked_resumes = []
    if request.method == 'POST':
        job_desc = request.form['job_description']
        files = request.files.getlist('resumes')
        paths = []
        for file in files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            paths.append(filepath)

        ranked_resumes = rank_resumes(paths, job_desc)

    return render_template('index.html', resumes=ranked_resumes)

if __name__ == '__main__':
    app.run(debug=True)
