import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Centralized configuration for the Medical RAG project.
    Sensitive information is loaded from environment variables (.env).
    """
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    HF_TOKEN = os.getenv('HF_TOKEN')

    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
    LANGSMITH_ENDPOINT = os.getenv('LANGSMITH_ENDPOINT', "https://api.smith.langchain.com")
    LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT', "skripsi_fixed")

    EMBEDDING_MODEL = "google/embeddinggemma-300m"
    EMBEDDING_DIM = 512
    BATCH_SIZE = 4
    MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

    COLLECTION_NAME = "2026_03_16__1147_enriched_medical_reports"

    # --- RAG Hyperparameters ---
    # Number of top documents to retrieve
    TOP_K = 3
    # LLM generation parameters
    TEMPERATURE = 0.1
    REPETITION_PENALTY = 1.1
    MAX_NEW_TOKENS = 512

    # --- File Paths ---
    # Path to the enriched medical documents (Pickle)
    PICKLE_PATH = "knowledge-base/2026_04_29__1545_medical_documents_enriched.pkl"
    DATA_FOLDER_PATH = 'knowledge-base/processed-1'
    RAW_DATA_PATH = 'knowledge-base/raw'
    PROCESSED_DATA_PATH = 'knowledge-base/processed'
    # Path to the golden dataset for evaluation
    GOLDEN_DATASET_PATH = "nutrition_qa_golden_dataset.json"

    @classmethod
    def to_dict(cls):
        """Converts config class attributes to a dictionary for compatibility."""
        return {
            'OPENAI_API_KEY': cls.OPENAI_API_KEY,
            'HF_TOKEN': cls.HF_TOKEN,
            'QDRANT_URL': cls.QDRANT_URL,
            'QDRANT_API_KEY': cls.QDRANT_API_KEY,
            'LANGSMITH_API_KEY': cls.LANGSMITH_API_KEY,
            'LANGSMITH_ENDPOINT': cls.LANGSMITH_ENDPOINT,
            'LANGSMITH_PROJECT': cls.LANGSMITH_PROJECT,
            'EMBEDDING_MODEL': cls.EMBEDDING_MODEL,
            'EMBEDDING_DIM': cls.EMBEDDING_DIM,
            'BATCH_SIZE': cls.BATCH_SIZE,
            'MODEL_ID': cls.MODEL_ID,
            'COLLECTION_NAME': cls.COLLECTION_NAME,
            'TOP_K': cls.TOP_K,
            'TEMPERATURE': cls.TEMPERATURE,
            'REPETITION_PENALTY': cls.REPETITION_PENALTY,
            'MAX_NEW_TOKENS': cls.MAX_NEW_TOKENS,
            'PICKLE_PATH': cls.PICKLE_PATH,
            'DATA_FOLDER_PATH': cls.DATA_FOLDER_PATH,
            'RAW_DATA_PATH': cls.RAW_DATA_PATH,
            'PROCESSED_DATA_PATH': cls.PROCESSED_DATA_PATH,
            'GOLDEN_DATASET_PATH': cls.GOLDEN_DATASET_PATH
        }
