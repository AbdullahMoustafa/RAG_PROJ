# RAG Project (Retrieval-Augmented Generation)

A Python-based Retrieval-Augmented Generation system using LangChain, Chroma, and OpenAI.

## Project Structure

```
RAG-Project/
├── src/                 # Source code
│   ├── __init__.py
│   └── rag.py          # Main RAG implementation
├── config/             # Configuration files
│   └── settings.py     # Settings and environment config
├── data/               # Data storage (documents, vector DB)
├── tests/              # Unit tests
├── main.py            # Entry point
├── requirements.txt   # Project dependencies
├── .env.example       # Environment variables template
├── .gitignore        # Git ignore file
└── README.md         # This file
```

## Setup Instructions

### 1. Clone and Navigate
```bash
cd d:\Linkedin\RAG\RAG-Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # If needed
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
```bash
copy .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### 6. Prepare Documents
Add your documents to the `data/` directory. Example:
```bash
echo "Your document content here" > data/sample.txt
```

### 7. Run the Application
```bash
python main.py
```

## Usage

Once running, you'll see an interactive prompt:
```
Ask a question: What is this document about?
```

The RAG system will retrieve relevant documents and generate an answer using OpenAI's LLM.

## Dependencies

- **langchain** - Framework for LLM applications
- **openai** - OpenAI API client
- **chromadb** - Vector database
- **sentence-transformers** - Embedding model
- **python-dotenv** - Environment variable management

## Configuration

Edit `config/settings.py` to customize:
- LLM model (default: gpt-3.5-turbo)
- Embedding model (default: all-MiniLM-L6-v2)
- Vector database path
- Chunk size for text splitting

## Troubleshooting

**Missing API Key:**
- Ensure you've set `OPENAI_API_KEY` in `.env`

**Vector DB Issues:**
- Delete `data/chroma_db/` and restart to reinitialize

**Virtual Environment Issues (Windows):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Next Steps

1. Customize `src/rag.py` for your specific use case
2. Add document loaders for PDFs, web pages, etc.
3. Implement caching and optimization
4. Add tests in `tests/` directory
5. Deploy using FastAPI or Flask for production
