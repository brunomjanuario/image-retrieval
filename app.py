from flask import Flask, request, jsonify
import retrieval

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)