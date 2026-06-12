import os

from dotenv import load_dotenv, find_dotenv
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv(find_dotenv(".env"))


print("Initializing components ...")
embeddings = OllamaEmbeddings(model="qwen3-embedding:4b", dimensions=1536)
llm = ChatOllama(model="qwen3.5:9b", temperature=0)
vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.environ["INDEX_NAME"])
retriever = vectorstore.as_retriever(search_kwargs={"k":3}) # Limit to top 3 relevant documents to get

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# ============================================================================
# IMPLEMENTATION 1: Without LCEL
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """Retrieval implementation without using LangChain Expression Language (LCEL).
    Manually retrieves documents, formats them, and generates a response.
    
    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additionl code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content

# ===============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ===============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe opreator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")
    # Query
    query = "What is Pinecone in machine learning?"

    # ============================================================================
    # Option 0: Raw invocation without RAG
    # ============================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ============================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ============================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ============================================================================
    # Option 2: Use implementation WITH LCEL
    # ============================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
    