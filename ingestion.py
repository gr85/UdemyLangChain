import asyncio
import os
import ssl
import certifi

from typing import Any, Dict, List
from dotenv import load_dotenv, find_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)


load_dotenv(find_dotenv(".env"))

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536, show_progress_bar=False, chunk_size=50, retry_min_seconds=10)
embeddings = OllamaEmbeddings(model="qwen3-embedding:4b", dimensions=1536)
# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestrate the entire process."""




if __name__ == "__main__":
    asyncio.run(main())