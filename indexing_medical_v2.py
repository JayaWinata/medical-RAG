import os
import glob
import pickle
import re
import datetime
from typing import Optional, List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import BitsAndBytesConfig
import torch

from openai import OpenAI
from pydantic import BaseModel, Field

from config.config import Config

# Load configuration from config.py
QDRANT_URL = Config.QDRANT_URL
QDRANT_API_KEY = Config.QDRANT_API_KEY
OPENAI_API_KEY = Config.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
HF_TOKEN = Config.HF_TOKEN
EMBEDDING_MODEL_ID = Config.EMBEDDING_MODEL
LLM_MODEL_ID = Config.MODEL_ID
DATA_FOLDER_PATH = Config.DATA_FOLDER_PATH
COLLECTION_NAME = Config.COLLECTION_NAME
PICKLE_BACKUP_PATH = Config.PICKLE_PATH

def get_section(md_text, start_pattern, max_length=5000):
    """
    Mengekstrak teks mulai dari pola judul yang ditemukan hingga batas karakter tertentu.
    Sangat kebal terhadap dokumen OCR yang formatnya berantakan.
    """
    if not md_text:
        return ""

    start_match = re.search(start_pattern, md_text, re.IGNORECASE | re.MULTILINE)
    if not start_match:
        return ""

    content_start = start_match.end()
    section_content = md_text[content_start:]
    section_content = section_content.strip()

    if max_length is not None and len(section_content) > max_length:
        section_content = section_content[:max_length]

    section_content = section_content.replace("|", " ")
    section_content = re.sub(r'-{2,}', ' ', section_content)
    section_content = re.sub(r' {2,}', ' ', section_content)
    section_content = "\n".join([line.rstrip() for line in section_content.splitlines()])
    section_content = section_content.strip()

    return section_content

CHAR_LENGTH = 6000
start_pattern = r"^(?:##\s*)?(?:BAB\s*[I1l\|]+\.?\s*)?(?:FORM\s*)?(?:NUTRITION(?:AL)?\s+CARE\s+PRO[CS]ESS?|NCP).*$"

class Antropometri(BaseModel):
    jenis_kelamin: str
    usia: str
    berat_badan: str
    tinggi_badan: str
    lingkar_kepala: Optional[str] = None
    lingkar_lengan_atas: Optional[str] = None
    indeks_massa_tubuh: Optional[str] = None

class DataMedis(BaseModel):
    anthropometric_data: Antropometri = Field(..., description="Objek data antropometri")
    medical_diagnosis: str = Field(..., description="Diagnosis medis")

    @property
    def anthropometric_string(self) -> str:
        data_dict = self.anthropometric_data.model_dump(exclude_none=True)
        formatted_list = [f"{k.replace('_', ' ').title()}: {v}" for k, v in data_dict.items()]
        return ", ".join(formatted_list)

def main():
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    print(f"Sedang membaca file dari: {DATA_FOLDER_PATH}/*.md ...")

    documents = []
    files = glob.glob(os.path.join(DATA_FOLDER_PATH, "*.md"))

    if not files:
        print("Tidak ada file .md ditemukan di folder tersebut!")
    else:
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                print(f"🔄 Processing: {filename}...", end=" ", flush=True)

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                text_input = get_section(text, start_pattern, max_length=CHAR_LENGTH)

                # Using structured output with gpt-4o-mini
                response = client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "Ekstrak informasi medis dari teks panjang yang diberikan"},
                        {"role": "user", "content": text_input},
                    ],
                    response_format=DataMedis,
                )

                extracted_data = response.choices[0].message.parsed

                if extracted_data:
                    meta_anthro = extracted_data.anthropometric_string
                    meta_diag = extracted_data.medical_diagnosis
                    meta_anthro_dict = extracted_data.anthropometric_data.model_dump(exclude_none=True)
                else:
                    meta_anthro = "Tidak Diketahui"
                    meta_diag = "Tidak Diketahui"
                    meta_anthro_dict = {}

                print(f" Diagnosis: {meta_diag}")

                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "anthropometric_assessment": meta_anthro,
                        "medical_diagnosis": meta_diag,
                        "anthropometric_data": meta_anthro_dict
                    }
                )
                documents.append(doc)
                print("✅ OK")

            except Exception as e:
                print(f"\n   ❌ Gagal load {filename}: {e}")

    if not documents:
        print("Tidak ada dokumen untuk diproses.")
        return

    print(f"\nSedang memecah {len(documents)} dokumen...")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    chunked_documents = []

    for doc in documents:
        md_chunks = text_splitter.split_text(doc.page_content)
        for chunk in md_chunks:
            chunk.metadata.update(doc.metadata)
        chunked_documents.extend(md_chunks)

    enriched_documents = []
    print("Sedang memodifikasi konten dokumen dengan metadata...")

    for i, doc in enumerate(chunked_documents):
        diagnosis = doc.metadata.get('medical_diagnosis', 'Tidak disebutkan')
        anthro = doc.metadata.get('anthropometric_assessment', 'Data antropometri tidak tersedia')

        new_content = f"""
INFO KLINIS PASIEN:
- Diagnosis Medis: {diagnosis}
- Assessment Fisik/Antropometri: {anthro}


ISI LAPORAN:
{doc.page_content}
"""
        new_metadata = doc.metadata.copy()
        new_metadata["chunk_id"] = i

        new_doc = Document(
            page_content=new_content.strip(),
            metadata=new_metadata
        )
        enriched_documents.append(new_doc)

    print(f"Selesai! {len(enriched_documents)} dokumen siap di-index.")

    print(f"\nMenyimpan backup chunks ke {PICKLE_BACKUP_PATH}...")
    os.makedirs(os.path.dirname(PICKLE_BACKUP_PATH), exist_ok=True)
    with open(PICKLE_BACKUP_PATH, "wb") as f:
        pickle.dump(enriched_documents, f)

    print("\nMemulai proses Embedding & Upload ke Qdrant...")

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
            "quantization_config": bnb_config,
            "torch_dtype": torch.float16,
        },
        encode_kwargs={
            "batch_size": 4,
            "show_progress_bar": True
        }
    )

    client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if not client_qdrant.collection_exists(COLLECTION_NAME):
        print(f"   - Membuat collection baru: {COLLECTION_NAME}")
        client_qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=Config.EMBEDDING_DIM if hasattr(Config, 'EMBEDDING_DIM') else 256,
                distance=models.Distance.COSINE
            )
        )

    vector_store = QdrantVectorStore.from_documents(
        documents=enriched_documents,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True
    )

    print(f"\nSELESAI! {len(enriched_documents)} chunks berhasil masuk ke Qdrant.")

if __name__ == '__main__':
    main()
