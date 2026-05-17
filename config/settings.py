import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Settings
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistral-7b-instruct-v0.1")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")

# Vector Database Settings
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
