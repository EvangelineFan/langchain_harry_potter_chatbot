from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
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

def get_formatted_docs_with_threshold(query: str, vector_store, relevance_threshold: float = 0.8) -> List[Document]:
    """
    Filter and format documents based on individual chunk relevance.
    Returns only chunks that meet the threshold, formatted with page numbers.
    """
    # Get more documents initially to have options for filtering
    docs_with_scores = vector_store.similarity_search_with_score(query, k=10)
    
    # Filter each chunk individually based on threshold
    relevant_docs = []
    for doc, score in docs_with_scores:
        if score <= relevance_threshold:  # Lower score = higher similarity
            page_num = doc.metadata.get('page', 'Unknown') # If "page" doesn’t exist in the metadata dictionary, defaults to "Unknown".
            formatted_doc = Document(
                page_content=f"[Page {page_num}]: {doc.page_content}",
                metadata={**doc.metadata, 'similarity_score': score}  # Keep original metadata + add score
            )
            relevant_docs.append(formatted_doc)
    
    # Return top 5 relevant documents (they're already sorted by similarity)
    return relevant_docs[:5]

# Prompt focuses on LLM strengths: reasoning, quote extraction, context interpretation
custom_prompt = ChatPromptTemplate.from_template("""
You are a Harry Potter assistant that answers questions using only the provided context.

Instructions:
1. If no context is provided, respond exactly with: "No evidence found in your document."
2. When context is available, always cite direct quotes and reference page numbers
3. Use conversation history to provide contextual responses
4. Never use knowledge outside the provided context

Previous conversation:
{chat_history}

Context from document:
{context}

Question: {question}

Answer:
""")

# Memory management handled by LangChain
memory = ConversationBufferWindowMemory(
    k=3,  # Keep last 3 exchanges
    memory_key="chat_history",
    return_messages=True
)

def format_chat_history(memory) -> str:
    """Convert memory messages to readable chat history string."""
    if not memory.chat_memory.messages:
        return "No previous conversation."
    
    history_parts = []
    messages = memory.chat_memory.messages[-6:]  # Last 3 exchanges
    
    # Pairs up messages like [Human, AI, Human, AI, Human, AI]
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            user_msg = messages[i].content
            ai_msg = messages[i + 1].content
            history_parts.append(f"Human: {user_msg}\nAssistant: {ai_msg}")
    
    return "\n\n".join(history_parts)

def ask_question(question: str):
    """
    Main function that orchestrates the RAG pipeline:
    1. Get relevant chunks (filtered by threshold)
    2. Format chat history
    3. Combine with system prompt
    4. Get LLM response
    5. Extract most relevant source
    """
    
    # Step 1: Get filtered relevant documents
    relevant_docs = get_formatted_docs_with_threshold(question, vector_store, relevance_threshold=0.8)
    
    # Step 2: Format chat history
    chat_history = format_chat_history(memory)
    
    # Step 3: Prepare context - STRICT GUARDRAIL
    if not relevant_docs:
        context = ""
        print(f"DEBUG: No relevant chunks found for: '{question}'")
    else:
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print(f"DEBUG: Using {len(relevant_docs)} chunks for context")
   
    # Step 4: Format prompt and get response
    formatted_prompt = custom_prompt.format(
        chat_history=chat_history,
        context=context,
        question=question
    )
    
    # Get LLM response
    from langchain.schema import HumanMessage
    response = chat_model.invoke([HumanMessage(content=formatted_prompt)])

    # CRITICAL GUARDRAIL CHECK: If no context but answer doesn't say "no evidence"
    if not relevant_docs and "No evidence found in your document" not in response.content:
        print(f"WARNING: LLM ignored 'no context' instruction!")
        print(f"LLM Response: {response.content}")
        # Force the correct response
        response.content = "No evidence found in your document."
    
    # Step 5: Update memory
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response.content)
    
    # Step 6: Prepare return with most relevant source
    if not relevant_docs:
        most_relevant_source = "No sources found"
    else:
        # Get the most relevant (first) document's page number
        first_doc = relevant_docs[0]
        page_num = first_doc.metadata.get('page', 'Unknown')
        most_relevant_source = f"Page {page_num}"
    
    return {
        "answer": response.content,
        "most_relevant_source": most_relevant_source,
        "context_chunks_used": len(relevant_docs)  # For debugging
    }

def debug_retrieval(query: str, threshold: float = 0.7):
    """Debug function to see what documents are being retrieved and filtered."""
    docs_with_scores = vector_store.similarity_search_with_score(query, k=10)
    
    print(f"\nDEBUG: Retrieval for '{query}' with threshold {threshold}")
    print("=" * 50)
    
    relevant_count = 0
    for i, (doc, score) in enumerate(docs_with_scores):
        status = "✓ INCLUDED" if score <= threshold else "✗ FILTERED OUT"
        if score <= threshold:
            relevant_count += 1
        
        page_num = doc.metadata.get('page', 'Unknown')
        print(f"{i+1}. Score: {score:.3f} | Page {page_num} | {status}")
        print(f"   Preview: {doc.page_content[:100]}...")
        print()
    
    print(f"Total relevant chunks (score ≤ {threshold}): {relevant_count}")
    print(f"Will use top 5 of these relevant chunks as context.")
    print("=" * 50)

# Test the system
if __name__ == "__main__":
    print("=== Harry Potter Q&A Assistant ===\n")
    
    # Test 1: Should find evidence
    print("Test 1: Known topic")
    result1 = ask_question("When did Harry first hear about Quidditch?")
    print(f"Q: When did Harry first hear about Quidditch?")
    print(f"A: {result1['answer']}")
    print(f"Most relevant source: {result1['most_relevant_source']}")
    print(f"Context chunks used: {result1['context_chunks_used']}\n")
    
    # Debug the first question to see filtering in action
    debug_retrieval("When did Harry first hear about Quidditch?", threshold=0.7)
    
    # Test 2: Should know "this sport" refers to Quidditch from conversation history
    print("Test 2: Memory test")
    result2 = ask_question("What equipment is used to play this sport?")
    print(f"Q: What equipment is used to play this sport?")
    print(f"A: {result2['answer']}")
    print(f"Most relevant source: {result2['most_relevant_source']}")
    print(f"Context chunks used: {result2['context_chunks_used']}\n")
    
    # Test 3: Should return "no evidence found"  
    print("Test 3: Test the guardrail")
    result3 = ask_question("Who is Viktor Krum and what's his achievement in Quidditch?")
    print(f"Q: Who is Viktor Krum and what's his achievement in Quidditch?")
    print(f"A: {result3['answer']}")
    print(f"Most relevant source: {result3['most_relevant_source']}")
    print(f"Context chunks used: {result3['context_chunks_used']}\n")
    
    # Test 4: Should find the Easter egg about Snitch
    print("Test 4: Easter egg test")
    result4 = ask_question("What should Harry be aware of when working with the latest Golden Snitches?")
    print(f"Q: What should he be aware of when working with the latest Golden Snitches?")
    print(f"A: {result4['answer']}")
    print(f"Most relevant source: {result4['most_relevant_source']}")
    print(f"Context chunks used: {result4['context_chunks_used']}")