import os
import re

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from tavily import TavilyClient


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES FIRST
# ============================================================

load_dotenv()


# ============================================================
# 2. CHECK API KEYS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from .env"
    )

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is missing from .env"
    )


# ============================================================
# 3. CREATE TAVILY CLIENT
# ============================================================

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


# ============================================================
# 4. TEMPERATURE TOOL
# ============================================================

@tool
def get_temperature(city: str) -> str:
    """
    Get the current temperature of a city.
    Returns a concise temperature value.
    """

    response = tavily_client.search(
        query=f"current temperature in {city} Celsius",
        search_depth="basic",
        max_results=3,
        include_answer=True,
    )

    # --------------------------------------------------------
    # Combine Tavily answer + search snippets
    # --------------------------------------------------------

    texts = []

    answer = response.get("answer")

    if answer:
        texts.append(answer)

    for result in response.get("results", []):
        content = result.get("content", "")

        if content:
            texts.append(content)

    combined_text = " ".join(texts)


    # --------------------------------------------------------
    # Prefer Celsius
    # --------------------------------------------------------

    celsius_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*°?\s*C",
        combined_text,
        re.IGNORECASE,
    )

    if celsius_match:

        temperature = celsius_match.group(1)

        return (
            f"Current temperature in {city}: "
            f"{temperature}°C"
        )


    # --------------------------------------------------------
    # Fallback to Fahrenheit
    # --------------------------------------------------------

    fahrenheit_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*°?\s*F",
        combined_text,
        re.IGNORECASE,
    )

    if fahrenheit_match:

        temperature = fahrenheit_match.group(1)

        return (
            f"Current temperature in {city}: "
            f"{temperature}°F"
        )


    return (
        f"Could not find the current temperature "
        f"for {city}."
    )


# ============================================================
# 5. MULTIPLICATION TOOL
# ============================================================

@tool
def multiply(
    a: float,
    b: float,
) -> float:
    """
    Multiply two numbers.
    """

    return a * b


# ============================================================
# 6. TOOL LIST
# ============================================================

tools = [
    get_temperature,
    multiply,
]


# ============================================================
# 7. CREATE GROQ MODEL
# ============================================================

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
)


# ============================================================
# 8. BIND TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(
    tools
)