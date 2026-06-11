from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient

load_dotenv(find_dotenv(".env"))

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query, num_results=3)


def main():
    print("Hello from langchain-course!")
    llm = ChatOllama(model="gpt-oss:20b", temperature=0)
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools)

    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})

    print(result)


if __name__ == "__main__":
    main()
