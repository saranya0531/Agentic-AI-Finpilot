# backend/agent.py

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from react_template import get_react_prompt_template
from tools.mytools import *
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# Choose ONE LLM (Groq is fast & free-tier friendly)
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Tools
tools = [
    add, subtract, multiply, divide, power,
    search, repl_tool,
    get_historical_price, get_current_price,
    get_company_info, schedule_task, check_system_time
]

# Prompt
prompt_template = get_react_prompt_template()

# Agent
agent = create_react_agent(llm, tools, prompt_template)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

def run_agent(user_input: str) -> str:
    try:
        response = agent_executor.invoke({"input": user_input})
        return f"<Response>{response['output']}</Response>"
    except Exception:
        return "<Response>Sorry, something went wrong.</Response>"
