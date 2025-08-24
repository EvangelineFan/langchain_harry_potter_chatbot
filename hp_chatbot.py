from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseRetriever
from typing import List
from langchain.schema import Document
from dotenv import load_dotenv
import os

# Setup
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(openai_api_key=api_key, temperature=0) # Temperature 0 means deterministic output, good for factual answers
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key) # Incurs small cost, overall a cheap and fast option for most RAG apps

# # Load and process documents
# loader = PyPDFLoader("HP1 Wiki (Enhanced).pdf") # Works well with clean, text-based PDFs; less well with badly scanned or formatted ones
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True) # Adds character position (as in the doc) to track where each chunk started in the original text
# chunks = text_splitter.split_documents(documents)
# vector_store = FAISS.from_documents(chunks, embeddings) # FAISS is free and stores it in local memory

# #To persist the vector store for examination, save it to disk
# vector_store.save_local("hp_faiss_index")
# #In case we need to load it from disk later
vector_store = FAISS.load_local("hp_faiss_index", embeddings, allow_dangerous_deserialization=True) # Only allowing dangerous deserialization here because we created this file

# ARCHITECTURAL DECISION:
# - Custom Retriever (Python): Handles relevance filtering + page number formatting
# - QA Chain: Handles LLM reasoning, memory management, prompt execution
# - Manual wrapper: Provides clean interface and result formatting

# class PageFormattedRetriever(BaseRetriever):
#     """
#     Custom retriever that handles two Python-specific tasks:
#     1. Relevance filtering (similarity threshold)
#     2. Page number formatting (data preprocessing)
#     """
    
#     def __init__(self, vector_store, relevance_threshold: float = 0.8):
#         super().__init__()  # Initialize the parent class
#         self.vector_store = vector_store
#         self.relevance_threshold = relevance_threshold
    
#     def get_relevant_documents(self, query: str) -> List[Document]: 
#         # Get documents with similarity scores
#         docs_with_scores = self.vector_store.similarity_search_with_score(query, k=5)
        
#         # Python handles relevance filtering (LLM can't see similarity scores)
#         if not docs_with_scores or docs_with_scores[0][1] > self.relevance_threshold: 
#             return []  # Empty context -> LLM will return "no evidence"
        
#         # Python handles formatting (ensures consistent page number display)
#         formatted_docs = []
#         for doc, score in docs_with_scores:
#             page_num = doc.metadata.get('page', 'Unknown') 
#             formatted_doc = Document(
#                 page_content=f"[Page {page_num}]: {doc.page_content}",
#                 metadata=doc.metadata
#             )
#             formatted_docs.append(formatted_doc)
        
#         return formatted_docs

def get_formatted_docs_with_threshold(query: str, vector_store, relevance_threshold: float = 0.8) -> List[Document]: # Document is a class from LangChain that represents a text chunk with metadata
    """
    Python function that handles two specific tasks:
    1. Relevance filtering (similarity threshold)
    2. Page number formatting (data preprocessing)
    """
    # Get documents with similarity scores
    docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
    
    # Python handles relevance filtering (LLM can't see similarity scores)
    if not docs_with_scores or docs_with_scores[0][1] > relevance_threshold: # If there are no documents OR the best document has a similarity score worse than 0.8, no good evidence. (Lower score = higher similarity)
        return []  # Empty list -> LLM will return "no evidence"
    
    # Python handles formatting (ensures consistent page number display)  
    formatted_docs = []
    for doc, score in docs_with_scores:
        page_num = doc.metadata.get('page', 'Unknown') # If "page" doesn’t exist in the metadata dictionary, defaults to "Unknown".
        formatted_doc = Document(
            page_content=f"[Page {page_num}]: {doc.page_content}",
            metadata=doc.metadata
        )
        formatted_docs.append(formatted_doc)
    
    return formatted_docs

# Prompt focuses on LLM strengths: reasoning, quote extraction, context interpretation
custom_prompt = ChatPromptTemplate.from_template("""
You are a Harry Potter assistant that answers questions using only the provided context.

Instructions:
1. If no context is provided, respond exactly with: "No evidence found in your document."
2. When context is available, always cite direct quotes and reference page numbers
3. Never use knowledge outside the provided context

Context from document:
{context}

Question: {question}

Answer:
""")

# Memory management handled by LangChain
memory = ConversationBufferWindowMemory(
    k=3,  # Keep last 3 exchanges
    memory_key="chat_history", # Variable name in prompt template
    return_messages=True, # Returns structured Message objects instead of plain strings
    input_key="query",  # THIS IS THE KEY FIX - tells memory which input to track
    output_key="result"  # THIS tells memory which output to track
)

# Create custom retriever and QA chain

qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True, # Includes the source chunks in the response object (doesn't automatically cite them in answers - need to format that ourselves)
    memory=memory, # Moved this outside of chain_type_kwargs
    chain_type_kwargs={
        "prompt": custom_prompt
    }
)

def ask_question(question: str):
    """
    Clean interface that leverages QA chain + provides formatted output
    
    This manual wrapper:
    - Provides consistent return format
    - Could add additional post-processing if needed
    - Maintains clean separation from internal chain logic
    """
    # Use our custom function to get formatted docs with threshold checking
    formatted_docs = get_formatted_docs_with_threshold(question, vector_store, relevance_threshold=0.8)
    # If no good docs, let the prompt handle it by passing the question with empty context
    if not formatted_docs:
        # Create a manual prompt with empty context to trigger "no evidence" response
        from langchain.schema import HumanMessage
        formatted_prompt = custom_prompt.format(
            context="",  # Empty context
            question=question,
            chat_history=memory.chat_memory.messages[-6:] if memory.chat_memory.messages else []
        )
        response = chat_model.invoke([HumanMessage(content=formatted_prompt)])
        
        # Update memory manually for this case
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(response.content)
        
        return {
            "answer": response.content,
            "source_documents": [],
            "num_sources": 0
        }
    # For good docs, use the standard QA chain
    result = qa_chain.invoke({"query": question})
    
    return {
        "answer": result["result"],
        "source_documents": result.get("source_documents", []),
        "num_sources": len(result.get("source_documents", []))
    }

# Alternative: Pure QA chain approach (using even less manual code)
# def ask_question_simple(question: str):
#     """Direct QA chain usage - minimal manual intervention"""
#     return qa_chain({"query": question})

# Test the system
if __name__ == "__main__":
    print("=== Harry Potter Q&A Assistant ===\n")
    
    # Test 1: Should find evidence with page numbers and quotes
    print("Test 1: Known topic")
    result1 = ask_question("What is Quidditch?")
    print(f"Q: What is Quidditch?")
    print(f"A: {result1['answer']}")
    print(f"Sources used: {result1['num_sources']}\n")
    
    # Test 2: Should return "no evidence found"  
    print("Test 2: General knowledge on topic but outside of document")
    result2 = ask_question("How many players are there in this sport?")
    print(f"Q: How many players are there in this sport?")
    print(f"A: {result2['answer']}")
    print(f"Sources used: {result2['num_sources']}\n")
    
    # Test 3: Should use conversation history
    print("Test 3: Follow-up question")
    result3 = ask_question("What is Harry's role in this sport?")
    print(f"Q: What is Harry's role in this sport?")
    print(f"A: {result3['answer']}")
    print(f"Sources used: {result3['num_sources']}\n")
    
    # Test 4: Another follow-up to test memory
    print("Test 4: Memory test")
    result4 = ask_question("What should he be aware of when working with the latest Snitch?")
    print(f"Q: What should he be aware of when working with the latest Snitch?")
    print(f"A: {result4['answer']}")
    print(f"Sources used: {result4['num_sources']}")