# Thờ Mẫu Agents - RAG API

An intelligent Retrieval-Augmented Generation (RAG) system for knowledge about Vietnamese Mother Goddess worship (Thờ Mẫu), using AI to create digital replicas of artisans in this field.

## Overview

This project builds a web API using FastAPI to provide conversational experiences with "AI replicas" of Thờ Mẫu artisans. The system uses:

- **ChromaDB**: Vector database for storing and searching knowledge
- **LlamaIndex**: RAG framework for indexing and retrieval
- **Claude (Anthropic)**: LLM for generating responses
- **Docling**: For processing and extracting content from PDFs
- **PostgreSQL**: Relational database for metadata and logs

## Key Features

- **AI Replica Chat**: Chat with Thờ Mẫu artisans in their characteristic style
- **Multi-tenant Knowledge Base**: Knowledge partitioned by individual artisans
- **Intelligent Rewriting**: Automatically rewrite questions based on conversation history
- **Fallback System**: Handle difficult questions by forwarding to other AIs or real artisans
- **PDF Ingestion**: Automatically process and index PDF documents
- **Memory System**: Store conversation history to maintain context

## Installation

### System Requirements
- Python 3.8+
- PostgreSQL database
- Anthropic API key

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the root directory:
```
SECRET_API_KEY=your_anthropic_api_key_here
DATABASE_URL=postgresql://user:password@localhost/dbname
```

### Initialize Database
```bash
python -c "from database import engine; from models import Base; Base.metadata.create_all(bind=engine)"
```

## Usage

### Run Server
```bash
uvicorn main:app --reload
```

The API will run at `http://localhost:8000`

### API Endpoints

#### POST `/api/chat`
Chat with an artisan's AI replica.

**Request Body:**
```json
{
  "session_id": "unique_session_id",
  "artisan_id": 1,
  "user_query": "What is Hau Dong?"
}
```

**Response:**
```json
{
  "response": "AI response...",
  "context_used": [...]
}
```

### Data Ingestion
To add new documents:
```python
from ingest import ingest_pdf
ingest_pdf("path/to/document.pdf", "Book Title")
```

## Project Structure

```
├── main.py              # FastAPI application and API endpoints
├── models.py            # SQLAlchemy models for database
├── database.py          # Database connection and session management
├── ingest.py            # Script to process and index PDF documents
├── ai_worker.py         # Background workers for AI processing
├── requirements.txt     # Python dependencies
├── chroma_db/           # ChromaDB vector database storage
├── data_sach/           # Directory containing PDF documents
└── import.ipynb         # Jupyter notebook for testing and development
```

## System Architecture

### AI Agents
- **AI A**: Main chatbot, answers based on available knowledge
- **AI B**: Handles difficult questions, rewrites prompts for artisans
- **AI C**: Interviews real artisans to collect new knowledge

### Database Schema
- `artisans`: List of artisans
- `documents`: Documents and metadata
- `document_chunks`: Text chunks that have been processed
- `chat_logs`: Conversation history
- `global_unanswered_questions`: Pool of unanswered questions
- `interview_queue`: Interview queue
- `artisan_answers`: Answers from artisans

## Development

### Adding New Artisans
1. Add record to `artisans` table
2. Upload their private documents with `owner_id`
3. Configure appropriate prompts

### Expanding Knowledge
- Upload new PDFs via API or `ingest.py` script
- Artisans can answer difficult questions through the interview system

## License

