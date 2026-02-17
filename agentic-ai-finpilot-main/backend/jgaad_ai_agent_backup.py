import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Create Gemini client
client = genai.Client(api_key=API_KEY)

# System instruction (financial advisor persona)
SYSTEM_INSTRUCTION = """
You are a knowledgeable personal financial advisor dedicated to helping individuals
navigate their financial journey.

Focus areas:
- Budgeting and expense tracking
- Investment strategies
- Retirement planning
- Debt management
- Tax planning
- Emergency funds
- Risk management and insurance

Provide practical, ethical, and realistic advice.
If research data is provided, use it carefully.
"""

# Generation configuration
GEN_CONFIG = types.GenerateContentConfig(
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
    system_instruction=SYSTEM_INSTRUCTION,
)

def jgaad_chat_with_gemini(query: str, research: str = "") -> str:
    """
    Agentic Gemini chat function for financial advice
    """

    prompt = f"""
    Research Context (if any):
    {research}

    User Question:
    {query}

    Respond with clear, structured financial advice.
    """

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
        config=GEN_CONFIG
    )

    return response.text


# ---------------- TEST ----------------
if __name__ == "__main__":
    test_query = "Should I invest in IT companies now?"
    print("Test Query:", test_query)

    answer = jgaad_chat_with_gemini(test_query)
    print("\nGemini Response:\n", answer)
