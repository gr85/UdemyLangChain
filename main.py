from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(".env"))

tavily = TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

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
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})

    print(result)
    print("="*50)
    print(result["messages"][-1].content)
    print("="*50)
    print(result["structured_response"])
    print("-"*50)



if __name__ == "__main__":
    main()
