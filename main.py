import torch
import datetime
import gc
import json
import os
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_qdrant import QdrantVectorStore
from tqdm import tqdm
from config.config import Config

from rag_pipeline import RAGBaseline, RAGFusion, RAGPipelineBase

if __name__ == '__main__':
    config = Config.to_dict()

    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    input_file = config["GOLDEN_DATASET_PATH"]
    output_file = f"{current_date}_rag_responses_dataset_k{config['TOP_K']}.jsonl"

    # SETUP ENVIRONMENT
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = config['LANGSMITH_ENDPOINT']
    os.environ["LANGCHAIN_PROJECT"] = config['LANGSMITH_PROJECT']
    os.environ["LANGCHAIN_API_KEY"] = config['LANGSMITH_API_KEY']
    os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']

    # Initialize Base Pipeline once to share LLM and VectorStore
    print("Menginisialisasi Model dan Vector Database...")
    base_pipeline = RAGPipelineBase(config=config)
    
    rag_baseline = RAGBaseline(config=config, llm=base_pipeline.llm, vectorstore=base_pipeline.vectorstore)
    rag_fusion = RAGFusion(config=config, llm=base_pipeline.llm, vectorstore=base_pipeline.vectorstore)
    print("Inisialisasi selesai!\n")

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Processing {len(dataset)} questions...")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(dataset):
            original_question = item['question']
            
            # Extract diagnosis for metadata filtering and query
            diagnosis = item['chunk_metadata'][0].get('medical_diagnosis', '')
            query_to_run = f"Diagnosis: {diagnosis}. Pertanyaan: {original_question}"
            filter_dict = {"medical_diagnosis": diagnosis} if diagnosis else None

            # Baseline execution
            baseline_data = {"answer": "ERROR", "ids": [], "contexts": [], "metadata": []}
            for attempt in range(3):
                try:
                    res_b = rag_baseline.run(query_to_run, filter_dict=filter_dict)
                    baseline_data = {
                        "answer": res_b['answer'],
                        "ids": res_b['ids'],
                        "contexts": [d.page_content for d in res_b['contexts']],
                        "metadata": [d.metadata for d in res_b['contexts']]
                    }
                    break
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n[Baseline OOM] Pertanyaan ke-{item.get('id', '?')} gagal. Melakukan pembersihan memori dan percobaan ulang ({attempt+1}/3)...")
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        print(f"Baseline error: {e}")
                        break

            # Fusion execution
            fusion_data = {"answer": "ERROR", "ids": [], "contexts": [], "metadata": []}
            for attempt in range(3):
                try:
                    res_f = rag_fusion.run(query_to_run, filter_dict=filter_dict)
                    fusion_data = {
                        "answer": res_f['answer'],
                        "ids": res_f['ids'],
                        "contexts": [d.page_content for d in res_f['contexts']],
                        "metadata": [d.metadata for d in res_f['contexts']]
                    }
                    break
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n[Fusion OOM] Pertanyaan ke-{item.get('id', '?')} gagal. Melakukan pembersihan memori dan percobaan ulang ({attempt+1}/3)...")
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        print(f"Fusion error: {e}")
                        break

            # Construct result
            result = {
                "question": original_question,
                "ground_truth": item['ground_truth'],
                "chunk_metadata": item['chunk_metadata'],
                "baseline": baseline_data,
                "fusion": fusion_data
            }

            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            del res_b, res_f
            gc.collect()
            torch.cuda.empty_cache()
