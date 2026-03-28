# Yashodeep's Portfolio Chatbot

A RAG-powered chatbot that answers questions about Yashodeep's skills, education, and experience.

## What it does

Ask the chatbot anything about Yashodeep's background, and it retrieves relevant information from a knowledge base to provide accurate answers.

## Tech Stack

- **FastAPI** - REST API framework
- **LangChain** - RAG orchestration
- **Groq** - LLM for intelligent responses
- **FAISS** - Vector similarity search
- **HuggingFace** - Embeddings generation

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**

   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY to .env
   ```

3. **Run the server:**

   ```bash
   uvicorn app:app --reload
   ```

4. **Chat with the bot:**

   ```bash
   curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Yashodeep'\''s educational background?"}'
   ```

## API Endpoints

- `GET /` - Welcome message
- `POST /chat` - Send a query to the chatbot

## Project Structure

- `app.py` - FastAPI application
- `main.py` - RAG chain setup
- `data.txt` - Knowledge base
- `faiss_index/` - Vector store

---

Built with ❤️
