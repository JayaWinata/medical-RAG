import os
import pickle
import datetime
import gc
from typing import Optional, List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import BitsAndBytesConfig
import torch

from config import Config

# Load configuration from config.py
QDRANT_URL = Config.QDRANT_URL
QDRANT_API_KEY = Config.QDRANT_API_KEY
HF_TOKEN = Config.HF_TOKEN
EMBEDDING_MODEL_ID = Config.EMBEDDING_MODEL
EMBEDDING_DIM = Config.EMBEDDING_DIM if hasattr(Config, 'EMBEDDING_DIM') else 256
BATCH_SIZE = Config.BATCH_SIZE if hasattr(Config, 'BATCH_SIZE') else 4
COLLECTION_NAME = Config.COLLECTION_NAME
PICKLE_BACKUP_PATH = Config.PICKLE_PATH

def main():
    # Performance optimization for PyTorch
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    print(f"🔄 Memuat dokumen dari {PICKLE_BACKUP_PATH}...")
    
    if os.path.exists(PICKLE_BACKUP_PATH):
        with open(PICKLE_BACKUP_PATH, "rb") as f:
            enriched_documents = pickle.load(f)
        print(f"✅ Berhasil memuat {len(enriched_documents)} dokumen.")
    else:
        print(f"❌ File {PICKLE_BACKUP_PATH} tidak ditemukan. Silakan jalankan indexing_medical_v2.py terlebih dahulu.")
        return

    print("\nMemulai proses Embedding & Upload ke Qdrant...")

    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={
            "device": "cuda",
            "trust_remote_code": True,
            "token": HF_TOKEN,
            "normalize_embeddings": True
        },
        encode_kwargs={
            "batch_size": BATCH_SIZE,
            "show_progress_bar": True
        }
    )

    # Clean up memory before indexing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if not client_qdrant.collection_exists(COLLECTION_NAME):
        print(f"   - Membuat collection baru: {COLLECTION_NAME}")
        client_qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE
            ),
            metadata={
                "embedding_model": EMBEDDING_MODEL_ID,
                "created_at": datetime.datetime.now().isoformat()
            }
        )

    print(f"🚀 Sedang mengupload {len(enriched_documents)} chunks ke Qdrant...")
    
    vector_store = QdrantVectorStore.from_documents(
        documents=enriched_documents,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
        batch_size=BATCH_SIZE
    )

    print(f"\n✨ SELESAI! {len(enriched_documents)} chunks berhasil masuk ke Qdrant.")

if __name__ == '__main__':
    main()
