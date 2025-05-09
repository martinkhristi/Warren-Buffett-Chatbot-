import streamlit as st
import os
import json
import yfinance as yf
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Setup ---
st.set_page_config(page_title="Warren Buffett Bot", layout="wide")
st.title("Warren Buffett Chatbot – Powered by Qwen3-32B via SambaNova ⚡")
st.caption("Ask me about investing, stocks, or market wisdom – in the style of Warren Buffett, running on Qwen3-32B.")

# --- Sidebar for API Key Input ---
st.sidebar.header("🔐 API Configuration")

sambanova_api_key = st.sidebar.text_input("SambaNova API Key", type="password")
serpapi_api_key = st.sidebar.text_input("SerpAPI API Key", type="password")

if sambanova_api_key:
    os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key

# --- Show API Status ---
st.sidebar.header("✅ API Status")
st.sidebar.success("SambaNova API Key Set" if sambanova_api_key else "Missing SambaNova API Key")
st.sidebar.success("SerpAPI Key Set" if serpapi_api_key else "Missing SerpAPI Key")

# --- LangChain Components ---
from langchain_sambanova import ChatSambaNovaCloud
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper

# --- Buffett Prompt ---
MODEL_NAME = "Qwen3-32B"
TEMPERATURE = 0.5
MEMORY_KEY = "chat_history"

BUFFETT_SYSTEM_PROMPT = """
You are a conversational AI assistant modeled after Warren Buffett, the legendary value investor.
You communicate with wisdom, patience, and long-term thinking. Stick to his investing principles:
- Value investing
- Margin of safety
- Moats
- Excellent management
- Understanding of the business
- Long-term perspective
- Simplicity
Use plain language, analogies, and Buffett-style quotes like:
“Price is what you pay. Value is what you get.”
Never provide speculative or financial advice. Instead, explain Buffett's perspective.
"""

# --- Tool 1: Stock Data Tool (Yahoo Finance) ---
@st.cache_data(show_spinner=False)
def get_stock_info(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or all(x is None for x in [info.get("regularMarketPrice"), info.get("currentPrice")]):
            hist = ticker.history(period="5d")
            if hist.empty:
                return f"Error: No data found for {symbol}."
            last_close = hist['Close'].iloc[-1]
            current_price = last_close
        else:
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")

        data = {
            "symbol": symbol,
            "companyName": info.get("longName", "N/A"),
            "currentPrice": current_price,
            "peRatio": info.get("trailingPE", "N/A"),
            "earningsPerShare": info.get("trailingEps", "N/A"),
            "marketCap": info.get("marketCap", "N/A"),
            "dividendYield": info.get("dividendYield", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "summary": info.get("longBusinessSummary", "N/A")[:500] + "..."
        }

        return json.dumps(data)
    except Exception as e:
        return f"Error retrieving stock info: {str(e)}"

stock_data_tool = Tool(
    name="get_stock_financial_data",
    func=get_stock_info,
    description="Fetches financial data for a stock symbol."
)

# --- Tool 2: News Search Tool ---
def create_news_search_tool(api_key):
    if api_key:
        try:
            params = {"engine": "google_news", "gl": "us", "hl": "en", "num": 5}
            search_wrapper = SerpAPIWrapper(params=params, serpapi_api_key=api_key)
            return Tool(
                name="search_stock_news",
                func=search_wrapper.run,
                description="Searches recent news about a stock symbol."
            )
        except Exception as e:
            return Tool(name="search_stock_news", func=lambda x: f"News error: {e}", description="Unavailable.")
    return Tool(name="search_stock_news", func=lambda x: "News search unavailable (no API key).", description="Unavailable.")

news_search_tool = create_news_search_tool(serpapi_api_key)
tools = [stock_data_tool, news_search_tool]

# --- Set Up LLM and Agent ---
if not sambanova_api_key:
    st.warning("Please enter your SambaNova API Key in the sidebar to start.")
    st.stop()

try:
    llm = ChatSambaNovaCloud(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        top_p=0.9,
        max_tokens=1024
    )

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=BUFFETT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    if 'agent_executor' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
        agent = create_openai_functions_agent(llm, tools, prompt_template)
        st.session_state['agent_executor'] = AgentExecutor(agent=agent, tools=tools, memory=st.session_state['memory'], verbose=True)

except Exception as e:
    st.error(f"Failed to initialize chatbot: {str(e)}")
    st.stop()

# --- Chat Loop ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Ask me anything about investing like Warren Buffett."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask Buffett Bot..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        with st.spinner("Buffett is pondering..."):
            start_time = time.time()
            response = st.session_state['agent_executor'].invoke({"input": prompt})
            end_time = time.time()

        response_time = round(end_time - start_time, 2)
        output = response.get('output', "Sorry, something went wrong.")
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.chat_message("assistant").write(output)

        st.info(f"⏱️ Response Time: {response_time} seconds")
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
        st.chat_message("assistant").write(f"Error: {str(e)}")

# --- Reset Chat Button ---
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()
