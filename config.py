import os
from dotenv import load_dotenv
from pathlib import Path

# اقرأ .env من نفس جذر المشروع
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "⚠️  GROQ_API_KEY not set!\n"
        "Please:\n"
        "1. Get your FREE key from: https://console.groq.com/\n"
        "2. Add it to .env file: GROQ_API_KEY=gsk_your_key_here\n"
    )

# Groq Model
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding Model ( runs locally)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_SIZE = 768

# Qdrant Settings
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'saudi_labor_law')

# Document Processing
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))

print("✅ Config loaded successfully!")
print(f"   Model: {GROQ_MODEL}")
print(f"   API Key: {GROQ_API_KEY[:20]}..." if GROQ_API_KEY else "   API Key: NOT SET")