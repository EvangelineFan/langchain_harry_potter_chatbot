from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseRetriever
from dotenv import load_dotenv
import os

# change directory to the current folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Specify chat model
chat_model = ChatOpenAI(openai_api_key=api_key, temperature=0) # Temperature 0 means deterministic output, good for factual answers

# Specify embedding model 
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", # Incurs small cost, overall a cheap and fast option for most RAG apps
    openai_api_key=api_key
)

# Load and chunk private doc
loader = PyPDFLoader("HP1 Wiki (Enhanced).pdf") # Works well with clean, text-based PDFs; less well with badly scanned or formatted ones
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  # Adds character position (as in the doc) to track where each chunk started in the original text
)
chunks = text_splitter.split_documents(documents)

# Embed and create vector store
vector_store = FAISS.from_documents(chunks, embeddings) # FAISS is free and stores it in local memory
# #To persist the vector store for examination, save it to disk
# vector_store.save_local("hp_faiss_index")
# #In case we need to load it from disk later
# vector_store = FAISS.load_local("hp_faiss_index", embeddings)

# Custom prompt template with guardrails
custom_prompt = ChatPromptTemplate.from_template("""
You are a Harry Potter assistant that ONLY answers questions using the provided context from the uploaded document.

IMPORTANT RULES:
1. If the answer cannot be found in the provided context, respond with "No evidence found in the uploaded document."
2. Always include direct quotes from the source material when possible
3. Always include the page number when citing information
4. Consider the conversation history for context
5. Do not use your general knowledge about Harry Potter - only use the provided context

Previous conversation:
{chat_history}

Context from uploaded document:
{context}

Human question: {question}

Assistant response (include quotes and page numbers):
""")

# Add memory to keep track of conversation history
memory = ConversationBufferWindowMemory(
    k=3, # Keeps last 3 exchanges, forgets older ones
    memory_key="chat_history", # Variable name in prompt template
    return_messages=True # Returns structured Message objects instead of plain strings
)

# Set up RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff", # Puts all chunks into one prompt, good for small docs
    retriever=vector_store.as_retriever(
        search_kwargs={
            "k": 5,  # Get more chunks for better context
            "score_threshold": 0.3  # Lower threshold to avoid missing relevant info
        }
    ),
    return_source_documents=True, # Includes the source chunks in the response object (doesn't automatically cite them in answers - need to format that ourselves)
    chain_type_kwargs={
        "prompt": custom_prompt,
        "memory": memory
    }
)

# Define main function to ask questions with enhanced context and guardrails
def ask_question(question: str):
    """Ask a question with enhanced context and guardrails"""
    
    # Get relevant chunks
    relevant_docs = vector_store.similarity_search_with_score(question, k=5)
    
    # Check if any chunks meet minimum relevance threshold
    if not relevant_docs or relevant_docs[0][1] > 0.8:  # If there are no documents OR the best document has a similarity score worse than 0.8, no good evidence. (Lower score = higher similarity)
        return "No evidence found in the uploaded document."
    
    # Prepare context with page numbers and quotes
    context_with_metadata = []
    for doc, score in relevant_docs:
        page_num = doc.metadata.get('page', 'Unknown')
        content = doc.page_content
        context_with_metadata.append(f"[Page {page_num}]: {content}")
    
    context_text = "\n\n".join(context_with_metadata)
    
    # Get conversation history
    chat_history = memory.chat_memory.messages[-6:] if memory.chat_memory.messages else []
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    
    # Format the prompt
    formatted_prompt = custom_prompt.format(
        context=context_text,
        question=question,
        chat_history=history_text
    )
    
    # Get response
    response = chat_model.invoke(formatted_prompt)
    
    # Save to memory
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response.content)
    
    return {
        "answer": response.content,
        "source_documents": [doc for doc, _ in relevant_docs[:3]],
        "relevance_scores": [score for _, score in relevant_docs[:3]]
    }

# Test the system
if __name__ == "__main__":
    # Test questions
    print("Testing guardrails...")
    result1 = ask_question("What is Quidditch?")  # Should find info
    print("Q: What is Quidditch?")
    print(f"A: {result1['answer']}\n")
    
    result2 = ask_question("What's Harry's favorite ice cream flavor?")  # Should return "no evidence"
    print("Q: What's Harry's favorite ice cream flavor?")
    print(f"A: {result2['answer']}\n")
    
    # Test conversation memory
    result3 = ask_question("Who else was there?")  # Should use previous context
    print("Q: Who else was there?")
    print(f"A: {result3['answer']}\n")

# result = chat_model.predict("Hi!")
# print(result)