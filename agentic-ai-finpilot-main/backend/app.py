from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import os

from agent import run_agent
from gemini_fin_path import finpilot_gemini_chat

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "FinPilot backend running"})

@app.route("/agent", methods=["POST"])
def agent():
    inp = request.form.get("input")
    if not inp:
        return jsonify({"error": "No input"}), 400

    output = run_agent(inp)
    match = re.search(r"<Response>(.*?)</Response>", output, re.DOTALL)

    final_answer = match.group(1).strip() if match else output
    return jsonify({"output": final_answer})

@app.route("/ai-financial-path", methods=["POST"])
def ai_financial_path():
    input_text = request.form.get("input", "")
    risk = request.form.get("risk", "conservative")

    response = finpilot_gemini_chat(input_text, risk)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
