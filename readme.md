# Harry Potter Q&A Chatbot

A document-based question-answering system built with LangChain that answers questions about Harry Potter and the Philosopher's Stone (summary by chapter from HP fan Wiki, with Easter Eggs mixed in) using only information from a provided document.

## Architecture Overview

```
┌─────────────┐    ┌────────┐    ┌─────────────┐    ┌─────────────────┐
│   PDF doc   │───▶│ Chunks │───▶│ Vector Store│───▶│ Relevant Chunks │
└─────────────┘    └────────┘    └─────────────┘    └─────────────────┘
                                                              │
                   ┌─────────────┐                           │
                   │   Question  │──────────────────────────▶│
                   └─────────────┘                           │
                           │                                 │
                   ┌─────────────┐                           │
                   │   Memory    │──────────────────────────▶│
                   └─────────────┘                           ▼
                           ▲                          ┌─────────────┐
                           │                          │  Guardrail  │
                           │                          └─────────────┘
                           │                                  │
                           │                                  ▼
                           │                           ┌─────────────┐
                           └───────────────────────────│     LLM     │
                                                       └─────────────┘
                                                               │
                                                               ▼
                                                    ┌─────────────────────┐
                                                    │ Answer incl. citation│
                                                    │    & page number     │
                                                    └─────────────────────┘
```

### Key Architectural Decisions

1. **Individual Chunk Filtering**: Each chunk is evaluated against the similarity threshold independently, preventing irrelevant chunks from being included just because one good chunk was found
2. **Separated Memory & Context**: Chat history and document context are handled as distinct inputs to the LLM
3. **Strict Guardrail Enforcement**: Post-LLM validation prevents hallucination when no relevant documents are found
4. **Direct LLM Integration**: Simplified pipeline with manual prompt formatting for full control over the process

## Features

- **Document-only answers**: Only provides information found in the source document
- **Similarity threshold filtering**: Filters out irrelevant results using vector similarity scores
- **Citations and page number**: Includes page references for all answers
- **Memory management**: Maintains conversation context for follow-up questions
- **Fallback handling**: Returns "No evidence found" when information isn't available in the document

## Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hp-chatbot.git
cd hp-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. Place your PDF document in the project directory and update the filename in the code if needed.

## Usage

### Basic Usage

Run the chatbot with the test examples:
```bash
python hp_chatbot.py
```

### Interactive Usage

You can modify the script to ask custom questions by calling:
```python
result = ask_question("Your question here")
print(result['answer'])
```

## Project Structure

```
hp-chatbot/
├── hp_chatbot.py          # Main chatbot script
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (not in repo)
├── .gitignore           # Git ignore file
├── hp_faiss_index/      # Vector store (generated)
│   ├── index.faiss
│   └── index.pkl
├── HP1 Wiki (Enhanced).pdf  # Source document (if included)
└── README.md            # This file
```

## How It Works

### Architecture

1. **Document Processing**: PDF is loaded and split into chunks using RecursiveCharacterTextSplitter
2. **Vector Storage**: Document chunks are embedded using OpenAI's text-embedding-3-small model and stored in FAISS
3. **Retrieval**: User queries are embedded and matched against stored chunks using similarity search
4. **Filtering**: Results below similarity threshold are filtered out
5. **Response Generation**: ChatGPT processes relevant chunks and generates answers with citations

### Key Components

- **Custom Retriever**: Handles relevance filtering and page number formatting
- **QA Chain**: Manages LLM reasoning and prompt execution
- **Memory Management**: Tracks conversation history for context-aware responses
- **Threshold Filtering**: Ensures only relevant information is used for answers

## Configuration

### Similarity Threshold
Adjust the relevance threshold in the `get_formatted_docs_with_threshold` function:
```python
relevance_threshold=0.8  # Lower = more strict, Higher = more permissive
```

### Individual Chunk Filtering
The system evaluates each chunk independently against the threshold:
```python
for doc, score in docs_with_scores:
    if score <= relevance_threshold:  # Only include truly relevant chunks
        # Process and format chunk
```

### Debug Mode
Use the debug function to see retrieval and filtering in action:
```python
debug_retrieval("Your question here", threshold=0.8)
```

## Example Interactions

### ✅ Successful Document-Based Answers
```
Q: When did Harry first hear about Quidditch?
A: Harry first heard about Quidditch when he met Oliver Wood, who explained the basics of the game to him, including the various positions on a standard Quidditch team and the role of the Seeker. This occurred after Harry received his Nimbus 2000 and was summoned to Quidditch practice by Professor McGonagall. (Page 11)
Most relevant source: Page 11
Context chunks used: 2
```

### ✅ Memory-Based Follow-up Questions
```
Previous: "What is Quidditch?"
Q: What equipment is used to play this sport?
A: The equipment used to play Quidditch includes broomsticks, Quaffles, Bludgers, and the Golden Snitch. Harry was introduced to these items during his first Quidditch practice session with Oliver Wood. (Page 11)
Most relevant source: Page 11
Context chunks used: 2
```

### ✅ Proper Guardrail Enforcement
```
Q: Who is Viktor Krum and what's his achievement in Quidditch?
A: No evidence found in your document.
Most relevant source: No sources found
Context chunks used: 0

Q: How many players are on each Quidditch team?
A: No evidence found in your document.
Most relevant source: No sources found
Context chunks used: 0
```

### 🎯 Easter Egg Detection
```
Q: What should he be aware of when working with the latest Golden Snitches?
A: According to the document, Wood mentions that the latest Golden Snitches are embedded with "micro-enchantments" that use "magical machine learning" to adapt their flight patterns based on the Seeker's skill level. (Page 12)
Most relevant source: Page 12
Context chunks used: 1
```

## Known Issues & Limitations

- **Chronological vs Similarity**: System prioritizes most similar chunks over chronologically first mentions
- **Page number accuracy**: Depends on PDF metadata during document loading
- **Threshold sensitivity**: May miss relevant information if threshold is too strict

## Troubleshooting

### "No evidence found" for known information
- Lower the `relevance_threshold` value (e.g., from 0.8 to 0.7)
- Check if the information exists in the document chunks using the debug functions

### Incorrect page numbers
- Verify PDF metadata during document loading
- Check the `page` field in document metadata

### API Errors
- Ensure your OpenAI API key is valid and has sufficient credits
- Check that the `.env` file is properly formatted

## Dependencies

- `langchain-openai`: OpenAI integration for LangChain
- `langchain-community`: Community extensions for document loading and vector stores
- `langchain-core`: Core LangChain functionality
- `faiss-cpu`: Vector similarity search
- `pypdf`: PDF document loading
- `python-dotenv`: Environment variable management

## Development

### Adding New Features

1. **Custom Retrievers**: Extend the `BaseRetriever` class for specialized filtering
2. **Different Document Types**: Add support for other document loaders

### Testing

The script includes built-in tests that run automatically. Add new test cases in the `if __name__ == "__main__":` section.

## License

This project is for educational purposes. Please ensure you have appropriate rights to use any documents you process.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Uses [OpenAI's GPT models](https://openai.com/)
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)