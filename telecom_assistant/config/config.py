# Configuration settings
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in parent directory
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# LLM Configuration
LLM_MODEL = "gpt-4"  # You can change to "gpt-3.5-turbo" for faster/cheaper responses
LLM_TEMPERATURE = 0.1  # Low temperature for consistent responses

# Database Configuration
DB_PATH = Path(__file__).parent.parent / 'data' / 'telecom.db'

# Document Storage
DOCUMENTS_PATH = Path(__file__).parent.parent / 'data' / 'documents'

# Validate API Key
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file in the project root with your OpenAI API key."
    )
