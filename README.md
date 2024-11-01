# AI Customer Support Agent with RAG

A Python application that demonstrates an intelligent customer support chatbot using Retrieval-Augmented Generation (RAG). The system uses a JSON dataset containing Q&A pairs about Thoughtful AI's services and agents. The application features history-aware retrieval, allowing for natural follow-up questions while maintaining conversation context.

## Features
- JSON data ingestion with ChromaDB vector storage
- Context-aware chat interface with conversation history
- Intelligent query reformulation using LangChain's rephrase prompts
- Local embeddings using Ollama
- Simple Streamlit interface for interaction

## Technologies Used
- Python 3.x
- LangChain
- Ollama (mxbai-embed-large model) for embeddings
- ChromaDB for vector storage
- OpenAI GPT-4-mini for query answering
- Streamlit for the chat interface

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
   Create a `.env` file in the root directory with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Install and start Ollama:
   - Follow instructions at [Ollama.ai](https://ollama.ai)
   - Pull the mxbai-embed-large model:
     ```bash
     ollama pull mxbai-embed-large
     ```

4. Ingest the data:
   ```bash
   cd python
   python ingest.py
   ```

5. Run the chat interface:
   ```bash
   cd python
   streamlit run streamlit_app.py
   ```


## How it Works

1. **Data Ingestion** (`ingest.py`):
   - Loads Q&A pairs from JSON
   - Generates embeddings using Ollama's mxbai-embed-large model
   - Stores vectors in local ChromaDB

2. **Retrieval** (`retriever.py`):
   - Uses history-aware retrieval for context maintenance
   - Implements RAG using LangChain's hub prompts
   - Combines OpenAI's GPT-4-mini for answer generation

3. **Chat Interface** (`streamlit_app.py`):
   - Simple Streamlit interface for Q&A
   - Maintains chat history
   - Displays conversation thread

## Why Local Vector Store?

This project intentionally uses ChromaDB as a local vector store for several reasons:

1. **Data Privacy**: All proprietary data remains local, as neither the embeddings generation nor vector similarity search requires cloud services.

2. **Local Embeddings**: 
   - Uses Ollama's mxbai-embed-large embeddings model deployed locally
   - State-of-the-art embedding quality without data leaving your system

3. **Complete Control**: 
   - Embeddings and vector store are maintained in `./chroma_db`
   - Full control over data storage and access patterns

While cloud solutions like Pinecone are available, they come with considerations:
- By default, Pinecone stores both embeddings and raw chunks
- Implementing more secure patterns (like storing only embeddings and using obfuscated paths for retrieval) requires manual implementation
- Data must be transmitted to and stored in the cloud

The current implementation prioritizes data privacy and local control while maintaining high performance through ChromaDB's efficient vector storage and retrieval capabilities.

## License
This project is licensed under the Apache License, Version 2.0 (APL 2.0). See the LICENSE file [here](LICENSE) for details.