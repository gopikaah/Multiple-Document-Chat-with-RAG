from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from functions import ask, generate_data_store, get_pdf_files

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    pdf_files = get_pdf_files()
    return render_template('index.html', pdf_files=pdf_files)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        print("No file part in the request")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)
    if file and file.filename.endswith('.pdf'):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(f"Saving file to {filename}")
        file.save(filename)
        generate_data_store()
        return redirect(url_for('index'))
    else:
        print("File is not a PDF")
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def invoke():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        response = ask(query)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
