# Thờ Mẫu RAG API (Vietnamese Spirituality Knowledge System)

## Overview

This is a sophisticated Retrieval-Augmented Generation (RAG) system designed for Vietnamese cultural and spiritual knowledge management, specifically focused on "Đạo Mẫu" (Mother Goddess worship tradition). It creates AI-powered virtual versions of master artisans/spiritual guides ("Sư phụ/Nghệ nhân") that users can interact with through a chat interface. The system uses a multi-AI orchestration approach to handle user queries, manage knowledge, and continuously learn from interactions with actual practitioners.

## Features

- **FastAPI Backend**: REST API server with WebSocket/streaming capabilities for chat interactions
- **Multi-AI Worker System**:
  - AI A (Artisan Avatar): Main chatbot embodying artisan personalities
  - AI B (Coordinator): Monitors and reformulates difficult questions for experts
  - AI C (Profiler): Analyzes responses to build authentic personality profiles
- **Document Ingestion Pipeline**: Processes PDF documents with OCR and chunking for efficient retrieval
- **PostgreSQL Database**: Structured storage for artisans, documents, chat logs, and knowledge base
- **Vector Search**: Qdrant-based semantic search for context-aware responses
- **Continuous Learning**: Captures unanswered questions and routes them to real experts
- **Cultural Authenticity**: Maintains Vietnamese cultural norms and artisan-specific communication styles

## Architecture

### Main Components

1. **FastAPI Backend (main.py)**: Handles chat requests, session management, and background logging
2. **Database Layer (database.py + models.py)**: PostgreSQL with tables for artisans, documents, chat logs, and question queues
3. **Document Ingestion (ingest.py)**: PDF processing with Docling OCR and chunking
4. **AI Workers**:
   - `ai_summary_worker.py`: Question coordination and reformulation
   - `ai_profiler_worker.py`: Personality profiling from artisan responses
5. **Data Import (import_questions.py)**: Loads pre-drafted question scenarios

### Data Structure

- **data_sach/**: Knowledge base containing 8 Vietnamese PDF files about Đạo Mẫu
- **questions.txt**: Pre-drafted question topics for interview scenarios
- Database tables: Artisans, Documents, DocumentChunks, ChatLogs, GlobalUnansweredQuestions, InterviewQueue, ArtisanAnswers, PreDraftedQuestion

## Dependencies

### Python Packages
- fastapi
- uvicorn
- python-multipart
- sqlalchemy
- psycopg2-binary
- llama-index
- llama-index-vector-stores-qdrant
- llama-index-embeddings-google-genai
- google-genai
- qdrant-client
- PyPDF2
- python-dotenv

### External Services
- PostgreSQL Database (Supabase or self-hosted)
- Qdrant Cloud (Vector database)
- Google Gemini API (LLM and embeddings)

## Installation

1. **Clone/Extract Repository**
   ```bash
   cd c:\Users\dotru\Hustle\Agents_tho_thanh_mau
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the project root:

```env
# Database Configuration
DB_USER=your_postgres_user
DB_PASSWORD=your_postgres_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name

# Google Gemini
GEMINI_API_KEY=your_google_gemini_api_key

# Qdrant Cloud
QDRANT_URL=https://your-qdrant-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

## Running the System

### Step 1: Initialize Database
```bash
python database.py
# Auto-creates all tables on first run
```

### Step 2: Ingest Knowledge Base PDFs
```bash
python ingest.py
# Processes PDFs from data_sach/ into vector database
```

### Step 3: Import Question Scenarios
```bash
python import_questions.py
# Loads questions from questions.txt
```

### Step 4: Start API Server
```bash
uvicorn main:app --reload
# Runs on http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Step 5: Run AI Workers (in separate terminals)

**Terminal 1 - AI B (Coordinator):**
```bash
python ai_summary_worker.py
# Monitors and processes questions
```

**Terminal 2 - AI C (Profiler):**
```bash
python ai_profiler_worker.py
# Analyzes responses and updates profiles
```

## API Usage Example

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "session_id": "session_abc",
    "artisan_id": 1,
    "user_query": "Hầu giá Cô Bơ cần chuẩn bị gì?"
  }'
```

## Workflow Example

1. User submits difficult question → Stored in `GlobalUnansweredQuestions`
2. AI B reformulates question → Creates interview tasks in `InterviewQueue`
3. Artisans respond → Responses stored in `ArtisanAnswers`
4. AI C analyzes responses → Updates artisan personality profiles
5. Future queries use updated profiles for authentic responses

## Project Structure

```
Agents_tho_thanh_mau/
├── main.py                          # FastAPI server & AI A
├── ai_summary_worker.py             # AI B (Coordinator)
├── ai_profiler_worker.py            # AI C (Profiler)
├── database.py                      # Database config
├── models.py                        # ORM models
├── ingest.py                        # PDF ingestion
├── import_questions.py              # Question loader
├── import.ipynb                     # Jupyter notebook
├── requirements.txt                 # Dependencies
├── questions.txt                    # Question scenarios
├── data_sach/                       # Knowledge PDFs
└── __pycache__/                     # Cache
```

## Key Design Features

- Knowledge hierarchy with public/private access control
- Multi-AI orchestration for separation of concerns
- Continuous learning from user interactions
- Cultural authenticity in AI responses
- Scalable vector search architecture
- Memory-efficient PDF processing

## Future Enhancements

- Multi-language support
- Audio transcription integration
- Real-time streaming
- Analytics dashboard
- Automated worker scheduling