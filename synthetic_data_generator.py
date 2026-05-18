import json
import random
import os
import pickle
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from config import Config

# Load environment variables
load_dotenv()

def generate_multi_context_qa(client, contexts, n_questions=2):
    """Generates multi-hop QA pairs using OpenAI gpt-4o-mini."""
    combined_context = "\n\n---\n\n".join(contexts)
    prompt = f"""
    Anda bertugas membuat dataset evaluasi berkualitas tinggi untuk sistem RAG medis.
    Gunakan HANYA informasi yang terdapat pada bagian "Konteks" di bawah ini.

    Instruksi:
    1. Buat {n_questions} pertanyaan multi-hop.
    2. Pertanyaan WAJIB menghubungkan minimal 2 bagian informasi yang berbeda dari konteks yang sama.
    3. Fokus pada narasi, intervensi, atau edukasi.
    4. DILARANG menanyakan diagnosis medis, data antropometri (BB/TB), atau angka laboratorium secara langsung.

    Keluaran WAJIB dalam format JSON:
    [
        {{"question": "...", "answer": "..."}}
    ]

    Konteks:
    {combined_context}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = json.loads(response.choices[0].message.content)
        
        # Handle various possible JSON structures returned by the LLM
        if isinstance(content, dict):
            for key in ['questions', 'data', 'qa_pairs', 'pairs']:
                if key in content: 
                    return content[key]
            # If it's a single pair dict, wrap it in a list
            if 'question' in content:
                return [content]
            return [content]
        return content
    except Exception as e:
        print(f"Error generating QA: {e}")
        return []

def build_grouped_docs(documents):
    """Groups LangChain documents by their source metadata."""
    grouped = {}
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown')
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(doc)
    return grouped

def build_dataset(grouped_docs, client, max_samples=50):
    """Builds the final dataset by iterating through grouped documents."""
    dataset = []
    sources = list(grouped_docs.keys())

    pbar = tqdm(total=max_samples, desc="Generating Valid Multi-Hop Data")
    while len(dataset) < max_samples:
        source = random.choice(sources)
        docs_in_source = grouped_docs[source]

        if len(docs_in_source) < 2:
            continue

        # Pick 2-3 chunks from the SAME source
        num_chunks = min(3, len(docs_in_source))
        selected_docs = random.sample(docs_in_source, num_chunks)
        selected_texts = [d.page_content for d in selected_docs]

        qa_pairs = generate_multi_context_qa(client, selected_texts)

        if not isinstance(qa_pairs, list):
            qa_pairs = [qa_pairs]

        for qa in qa_pairs:
            if len(dataset) >= max_samples:
                break
            if not qa or 'question' not in qa:
                continue

            dataset.append({
                "question": qa["question"],
                "ground_truth": qa.get("answer") or qa.get("ground_truth"),
                "gold_contexts": selected_texts,
                "chunk_metadata": [d.metadata for d in selected_docs]
            })
            pbar.update(1)

    pbar.close()
    return dataset

def save_dataset(dataset, path):
    """Saves the generated dataset to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dataset)} samples to {path}")

def main():
    # Setup OpenAI client
    if not Config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in configuration.")
        return
        
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    PICKLE_PATH = Config.PICKLE_PATH
    OUTPUT_PATH = Config.GOLDEN_DATASET_PATH

    if not os.path.exists(PICKLE_PATH):
        print(f"Error: Pickle file {PICKLE_PATH} not found.")
        return

    print(f"Loading documents from {PICKLE_PATH}...")
    with open(PICKLE_PATH, 'rb') as f:
        documents = pickle.load(f)

    print("Grouping documents by source...")
    grouped = build_grouped_docs(documents)

    print(f"Starting generation for {50} samples...")
    # Using 50 samples as per notebook logic, but could be configurable
    final_dataset = build_dataset(grouped, client, max_samples=50)

    save_dataset(final_dataset, OUTPUT_PATH)

if __name__ == "__main__":
    main()
