from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages.ai import AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.environ.get('GROQ_API_KEY')
SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')


# Initialize LLMs
deepseek_llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=2048,  # Set explicit value
    timeout=None,
    max_retries=2,
)

groq_llm = ChatGroq(
    temperature=0,
    model_name="llama-3.2-3b-preview"
)


# Initialize Search Tool
params = {
    "engine": "google",
    "gl": "us",
    "hl": "en",
}
search_wrapper = SerpAPIWrapper(params=params)  # Search tool

# Convert to a valid tool
search_tool = Tool(
    name="Search",
    func=search_wrapper.run,  # Ensure it is a callable function
    description="Searches the web using SerpAPI. Provide a query to get real-time search results."
)


# Define Prompt Template
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a news expert"),
    ("user", "{messages}")
])


def getting_respose_from_agent(llm_name, query, allowed_search, system_prompt, provider):

    if provider.lower() == 'groq':
        llm = ChatGroq(model=llm_name)
    else:
        llm = ChatDeepSeek(model=llm_name,
                           temperature=0,
                           max_tokens=2048,  # Set explicit value
                           timeout=None,
                           max_retries=2,)

    # creating search tool
    params = {
        "engine": "google",
        "gl": "us",
        "hl": "en",
    }

    # Initialize Search Tool
    search_wrapper = SerpAPIWrapper(params=params)
    # Convert to a valid tool
    search_tool = Tool(
        name="Search",
        func=search_wrapper.run,  # Ensure it is a callable function
        description="Searches the web using SerpAPI. Provide a query to get real-time search results."
    )
    # if allowed_search condition
    tools = [search_tool] if allowed_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools,  # Use wrapped tool
        prompt=system_prompt  # Use formatted prompt
    )

    # Query
    # groq expects role:... ,content:....
    inputs = {"messages": [{"role": "user", "content": query}]}
    response = agent.invoke(inputs)
    messages = response.get("messages")
    ai_messages = [
        message.content for message in messages if isinstance(message, AIMessage)]
    # print(ai_messages[-1])
    return ai_messages[-1]

# Ensure correct key for output

# getting_respose_from_agent('qwen-qwq-32b','where is india',False,' You are a news expert. If you cannot find the answer, respond with "\I don\'t know.\"','groq')
