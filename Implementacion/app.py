from flask import Flask, request, jsonify, render_template
from detect import detect_objects

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Realiza la detección de objetos
    result = detect_objects(file)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
