from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("message", "")
    if not query:
        return jsonify({"response": "No input received."})

    from main import handle_query
    reply = handle_query(query)
    return jsonify({"response": reply})

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"response": "No image file provided."})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"response": "No image file selected."})

    file_path = "temp_uploaded_image.jpg"
    file.save(file_path)

    from image import analyze_image_text_and_faces
    result = analyze_image_text_and_faces(image_path=file_path)

    if os.path.exists(file_path):
        os.remove(file_path)

    return jsonify({"response": result})

@app.route("/songs", methods=["GET"])
def get_songs():
    from Automation import list_songs
    songs = list_songs()
    return jsonify({"response": songs})

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"response": "No prompt provided for image generation."})

    from image import text_to_image
    result = text_to_image(prompt)
    return jsonify({"response": result})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Pichu is running"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=True)
