"""Configuration settings for the AI Interview Analyzer."""
import os
from dotenv import load_dotenv

load_dotenv()

# Free Model Configuration (no API keys needed!)
USE_FREE_MODELS = True  # Set to False if you want to use OpenAI

# Whisper Configuration (Local, Free)
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large (base is good balance)

# Ollama Configuration (Local, Free LLM)
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2"  # Options: llama3.2, llama3.1, mistral, etc.

# Embedding Model (Local, Free)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Scoring Configuration
SCORING_SCALE = (1, 10)  # Min and max scores

# OpenAI Configuration (Optional - only if USE_FREE_MODELS = False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

