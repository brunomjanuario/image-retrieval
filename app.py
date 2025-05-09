from flask import Flask, request, jsonify, send_file, abort, send_from_directory
from flask_cors import CORS
import retrieval
import os

IMAGE_FOLDER = 'images'

app = Flask(__name__)
CORS(app)

@app.route('/search', methods=['POST'])
def search_images():

    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        results = retrieval.retrieval_images(prompt)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/image', methods=['GET'])
def get_image():
    filename = request.args.get('filename')  # get from query string

    if not filename:
        return abort(400, description="Missing 'filename' parameter")

    image_path = os.path.join(IMAGE_FOLDER, filename)

    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return abort(404, description="Image not found")


if __name__ == '__main__':
    app.run(debug=True, port=5000)