# currency_agent/agent.py
import logging
import os
from dotenv import load_dotenv

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams
from fastapi.staticfiles import StaticFiles   # ✅ Import StaticFiles

# --- Load .env variables ---
load_dotenv()
assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY not set!"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")

# --- Logging ---
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- System instruction ---
SYSTEM_INSTRUCTION = (
    "You are a friendly currency assistant. "
    "Use the 'get_exchange_rate' tool to fetch accurate exchange rates. "
    "Always provide answers in full sentences. "
    "If the user asks anything outside currency conversion, politely reply that you can only help with currency-related questions."
)

# --- LiteLLM for Groq ---
GROQ_MODEL = "groq/llama-3.3-70b-versatile"
logger.info(f"✅ Using Groq LiteLLM model: {GROQ_MODEL}")
lite_llm_model = LiteLlm(model=GROQ_MODEL)

# --- Create ADK Agent ---
root_agent = Agent(
    name="currency_agent",
    model=lite_llm_model,
    description="An AI assistant for currency conversion with tool integration",
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(url=MCP_SERVER_URL)
        )
    ],
)


# --- Convert to A2A-compatible app ---
a2a_app = to_a2a(root_agent, port=10000) 
logger.info("✅ Currency agent ready and A2A-compatible.")

