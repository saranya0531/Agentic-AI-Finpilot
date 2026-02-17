import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """
You are FinPilot, an AI-powered personal financial advisor.
Provide clear, practical, and ethical financial guidance.
"""

GEN_CONFIG = types.GenerateContentConfig(
    temperature=0.7,
    top_p=0.9,
    max_output_tokens=4096,
    system_instruction=SYSTEM_PROMPT,
)

def finpilot_gemini_chat(query: str, context: str = "") -> str:
    prompt = f"""
Context:
{context}

User Query:
{query}

Respond as FinPilot.
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
        config=GEN_CONFIG
    )

    return response.text
