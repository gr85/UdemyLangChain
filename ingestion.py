import os

from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv(find_dotenv(".env"))

def main():
    print("Ingesting")
    loader = TextLoader("/home/guiuriera/Documents/Courses/Udemy/LangChain_AgenticAI/langchain-course/mediumblog1.txt")
    document = loader.load()

    print("splitting ...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b", dimensions=1536)

    print("ingesting ...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])

    print("finish")

if __name__ == "__main__":
    main()
