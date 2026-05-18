import pandas as pd
import json
import os
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from config import Config

def load_jsonl(path):
    """Loads a JSONL file into a pandas DataFrame."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

def extract_final_answer(full_text):
    """Trims LLM prompt and only takes the final answer."""
    if not isinstance(full_text, str):
        return ""
    
    # Token penanda di mana asisten mulai menjawab
    split_token = "<|im_start|>assistant\n"
    
    if split_token in full_text:
        # Ambil semua teks setelah token asisten
        return full_text.split(split_token)[-1].strip()
    return full_text.strip()

def calc_retrieval_metrics(row, retrieved_col, k=3):
    """Calculates Precision@K, Recall@K, and MRR@K."""
    retrieved_ids = row[retrieved_col][:k] # Potong hasil sesuai K
    ground_truth_ids = row['ground_truth_ids']
    
    gt_set = set(ground_truth_ids)
    
    # Deteksi mana yang "Hit" (Benar) di urutan top K
    hits = [1 if doc_id in gt_set else 0 for doc_id in retrieved_ids]
    
    # 1. Recall@K = (Jumlah Benar) / (Total Kunci Jawaban)
    recall = sum(hits) / len(gt_set) if len(gt_set) > 0 else 0.0
    
    # 2. Precision@K = (Jumlah Benar) / K
    precision = sum(hits) / k if k > 0 else 0.0
    
    # 3. MRR@K = 1 / (Ranking pertama yang benar)
    mrr = 0.0
    for i, hit in enumerate(hits):
        if hit == 1:
            mrr = 1.0 / (i + 1)
            break
            
    return pd.Series([precision, recall, mrr])

def main():
    # Setup OpenAI API Key
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
    
    print("Initializing Evaluator LLM and Embeddings...")
    langchain_llm = ChatOpenAI(model="gpt-4o-mini")
    langchain_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    evaluator_llm = LangchainLLMWrapper(langchain_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    K_VALUE = Config.TOP_K
    # In the notebook this was hardcoded, using the notebook value for default if config is different
    # But usually config should be the source of truth.
    FILEPATH = f"knowledge-base/20260429_1829_rag_responses_dataset_k3.jsonl"
    
    if not os.path.exists(FILEPATH):
        print(f"Error: File {FILEPATH} not found.")
        return

    print(f"1. Loading evaluation data from {FILEPATH}...")
    df = load_jsonl(FILEPATH)
    print(f"Total rows loaded: {len(df)}\n")

    # Extract Ground Truth IDs from metadata
    df['ground_truth_ids'] = df['chunk_metadata'].apply(
        lambda x: [meta['chunk_id'] for meta in x] if isinstance(x, list) else []
    )

    # Extract Retrieval IDs from metadata
    df['baseline_retrieved_ids'] = df['baseline_contexts_metadata'].apply(
        lambda x: [meta['chunk_id'] for meta in x] if isinstance(x, list) else []
    )
    df['fusion_retrieved_ids'] = df['fusion_contexts_metadata'].apply(
        lambda x: [meta['chunk_id'] for meta in x] if isinstance(x, list) else []
    )

    # Clean answers
    print("Cleaning answers from prompts...")
    df['clean_baseline_answer'] = df['baseline_answer'].apply(extract_final_answer)
    df['clean_fusion_answer'] = df['fusion_answer'].apply(extract_final_answer)

    # 2. Calculate Retrieval Metrics
    print(f"2. Calculating Retrieval Metrics for K={K_VALUE}...")
    
    # Baseline Metrics
    df[['baseline_Precision@K', 'baseline_Recall@K', 'baseline_MRR@K']] = df.apply(
        lambda row: calc_retrieval_metrics(row, 'baseline_retrieved_ids', k=K_VALUE), axis=1
    )

    # Fusion Metrics
    df[['fusion_Precision@K', 'fusion_Recall@K', 'fusion_MRR@K']] = df.apply(
        lambda row: calc_retrieval_metrics(row, 'fusion_retrieved_ids', k=K_VALUE), axis=1
    )

    # 3. Prepare data for Ragas evaluation
    print("\n3. Preparing data for Ragas evaluation (LLM-as-a-Judge)...")

    # Format Dataset for Ragas Baseline
    dataset_baseline = Dataset.from_dict({
        "question": df["question"].tolist(),
        "answer": df["clean_baseline_answer"].tolist(),
        "contexts": df["baseline_retrieved_contexts"].tolist(),
        "ground_truth": df["ground_truth"].tolist()
    })

    # Format Dataset for Ragas Fusion
    dataset_fusion = Dataset.from_dict({
        "question": df["question"].tolist(),
        "answer": df["clean_fusion_answer"].tolist(),
        "contexts": df["fusion_retrieved_contexts"].tolist(),
        "ground_truth": df["ground_truth"].tolist()
    })

    print("\n> Running Ragas Evaluation for BASELINE...")
    ragas_result_baseline = evaluate(
        dataset_baseline,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\n> Running Ragas Evaluation for FUSION...")
    ragas_result_fusion = evaluate(
        dataset_fusion,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\n" + "="*50)
    print(f"FINAL EVALUATION RESULTS (Top-K = {K_VALUE})")
    print("="*50)

    df_ragas_baseline = ragas_result_baseline.to_pandas()
    df_ragas_fusion = ragas_result_fusion.to_pandas()

    # Calculate means
    baseline_faithfulness_mean = df_ragas_baseline['faithfulness'].mean()
    baseline_answer_rel_mean = df_ragas_baseline['answer_relevancy'].mean()

    fusion_faithfulness_mean = df_ragas_fusion['faithfulness'].mean()
    fusion_answer_rel_mean = df_ragas_fusion['answer_relevancy'].mean()

    summary_df = pd.DataFrame({
        "Metrik": [
            f"Context Recall@{K_VALUE}", 
            f"Context Precision@{K_VALUE}", 
            f"MRR@{K_VALUE}", 
            "Faithfulness (Ragas)", 
            "Answer Relevancy (Ragas)"
        ],
        "RAG Baseline": [
            df['baseline_Recall@K'].mean(),
            df['baseline_Precision@K'].mean(),
            df['baseline_MRR@K'].mean(),
            baseline_faithfulness_mean,
            baseline_answer_rel_mean
        ],
        "RAG Fusion": [
            df['fusion_Recall@K'].mean(),
            df['fusion_Precision@K'].mean(),
            df['fusion_MRR@K'].mean(),
            fusion_faithfulness_mean,
            fusion_answer_rel_mean
        ]
    })

    print(summary_df.to_string(index=False))

    # 4. Save results
    output_file = f'evaluation_@{K_VALUE}.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary results saved to {output_file}")

if __name__ == "__main__":
    main()
