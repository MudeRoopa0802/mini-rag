from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Mini RAG Backend Running 🚀"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")

    # Dummy response (replace with your RAG logic)
    answer = f"Answer for: {question}"

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
