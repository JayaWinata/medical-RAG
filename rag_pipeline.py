import torch
from qdrant_client import QdrantClient
from operator import itemgetter
from langchain_core.load import dumps, loads
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from tqdm import tqdm
import os

class RAGBaseline:
    def __init__(self, config):
        self.config = config
        self.setup_environment()
        self.setup_vectorstore()
        self.setup_llm()
        self.setup_retriever()
        self.setup_generation_chain() # Diubah dari setup_rag_chain

    def setup_environment(self):
        # ... (Sama seperti sebelumnya) ...
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = self.config['LANGSMITH_ENDPOINT']
        os.environ["LANGCHAIN_PROJECT"] = self.config['LANGSMITH_PROJECT']
        os.environ["LANGCHAIN_API_KEY"] = self.config['LANGSMITH_API_KEY']
        os.environ["OPENAI_API_KEY"] = self.config['OPENAI_API_KEY']
        os.environ["QDRANT_API_KEY"] = self.config['QDRANT_API_KEY']
        os.environ["QDRANT_URL"] = self.config['QDRANT_URL']
        os.environ["HF_TOKEN"] = self.config['HF_TOKEN']

    def setup_vectorstore(self):
        # ... (Sama seperti sebelumnya) ...
        self.client = QdrantClient(url=self.config['QDRANT_URL'], api_key=self.config['QDRANT_API_KEY'])
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config['EMBEDDING_MODEL'],
            model_kwargs={'device': 'cuda', 'token': self.config['HF_TOKEN']}
        )
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.config['COLLECTION_NAME'],
            embedding=self.embedding_model,
            content_payload_key="full_text",
        )

    def setup_retriever(self):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def setup_llm(self):
        # ... (Sama seperti sebelumnya) ...
        tokenizer = AutoTokenizer.from_pretrained(self.config['MODEL_ID'])
        model = AutoModelForCausalLM.from_pretrained(
            self.config['MODEL_ID'],
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=True
        )
        text_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            repetition_penalty=1.1
        )
        self.llm = HuggingFacePipeline(pipeline=text_pipe)

    def setup_generation_chain(self):
        """Hanya menyiapkan chain untuk GENERASI jawaban, bukan retrieval"""
        template = """<|im_start|>system
        Anda adalah asisten AI yang efisien untuk analisis laporan medis.
        Jawablah pertanyaan HANYA berdasarkan konteks yang diberikan.

        ATURAN PENTING:
        1. Jawab dengan SINGKAT, PADAT, dan LANGSUNG pada intinya.
        2. DILARANG mengulang-ulang kalimat atau mengulang pertanyaan user.
        3. Jangan menambahkan basa-basi seperti "Berdasarkan konteks yang diberikan...". Langsung berikan jawabannya.
        4. Jika informasi tidak ada di konteks, katakan "Informasi tidak ditemukan".
        <|im_end|>
        <|im_start|>user
        KONTEKS:
        {context}

        PERTANYAAN:
        {question}
        <|im_end|>
        <|im_start|>assistant
        """
        self.prompt = ChatPromptTemplate.from_template(template)

        # Chain ini mengharapkan input dict {'context': str, 'question': str}
        self.generation_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run(self, query):
        """
        Menjalankan pipeline secara bertahap agar bisa mengembalikan dokumen konteks.
        """
        # 1. Retrieve (Dapatkan Objek Dokumen Asli)
        retrieved_docs = self.retriever.invoke(query)
        
        # 2. Format Context ke String
        formatted_context = self.format_docs(retrieved_docs)
        
        # 3. Generate Answer
        answer = self.generation_chain.invoke({
            "context": formatted_context, 
            "question": query
        })
        
        # 4. Return Paket Lengkap
        return {
            "answer": answer,
            "contexts": retrieved_docs # List of Document objects (penting untuk evaluasi)
        }

class RAGFusion:
    def __init__(self, config):
        self.config = config
        self.setup_environment()
        self.setup_vectorstore()
        self.setup_llm()
        self.setup_retriever()
        self.setup_fusion_pipeline()
        self.setup_generation_chain() # Diubah dari setup_rag_chain

    def setup_environment(self):
        # ... (Sama seperti sebelumnya) ...
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = self.config['LANGSMITH_ENDPOINT']
        os.environ["LANGCHAIN_PROJECT"] = self.config['LANGSMITH_PROJECT']
        os.environ["LANGCHAIN_API_KEY"] = self.config['LANGSMITH_API_KEY']
        os.environ["OPENAI_API_KEY"] = self.config['OPENAI_API_KEY']
        os.environ["QDRANT_API_KEY"] = self.config['QDRANT_API_KEY']
        os.environ["QDRANT_URL"] = self.config['QDRANT_URL']
        os.environ['HF_TOKEN'] = self.config['HF_TOKEN']

    def setup_vectorstore(self):
        # ... (Sama seperti sebelumnya) ...
        self.client = QdrantClient(url=self.config['QDRANT_URL'], api_key=self.config['QDRANT_API_KEY'])
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config['EMBEDDING_MODEL'],
            model_kwargs={'device': 'cuda', 'token': self.config['HF_TOKEN']}
        )
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.config['COLLECTION_NAME'],
            embedding=self.embedding_model,
            content_payload_key="full_text",
        )

    def setup_retriever(self):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def setup_llm(self):
        # ... (Sama seperti sebelumnya) ...
        tokenizer = AutoTokenizer.from_pretrained(self.config['MODEL_ID'])
        model = AutoModelForCausalLM.from_pretrained(
            self.config['MODEL_ID'],
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=True
        )
        text_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            repetition_penalty=1.1
        )
        self.llm = HuggingFacePipeline(pipeline=text_pipe)

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        # ... (Sama seperti sebelumnya) ...
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
        
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return [doc for doc, _ in reranked_results]

    def setup_fusion_pipeline(self):
        # ... (Sama seperti sebelumnya) ...
        template_query_gen = """<|im_start|>system
        Anda adalah asisten AI yang membantu mengoptimalkan pencarian database medis.
        Tugas Anda adalah membuat 3 variasi pertanyaan pencarian yang berbeda berdasarkan pertanyaan awal pengguna.
        Variasi ini bertujuan untuk menangkap berbagai aspek dari masalah medis yang ditanyakan.

        Format Output: Hanya tuliskan 3 pertanyaan, dipisahkan dengan baris baru. Jangan pakai nomor.
        <|im_end|>
        <|im_start|>user
        Pertanyaan Asli: {question}
        <|im_end|>
        <|im_start|>assistant
        """
        prompt_rag_fusion = ChatPromptTemplate.from_template(template_query_gen)
        
        self.generate_queries = (
            prompt_rag_fusion
            | ChatOpenAI(temperature=0.2)
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
        )

        self.retrieval_chain_rag_fusion = (
            self.generate_queries 
            | self.retriever.map() 
            | self.reciprocal_rank_fusion
        )

    def setup_generation_chain(self):
        """Hanya setup LLM Generation"""
        template = """<|im_start|>system
        Anda adalah asisten AI yang efisien untuk analisis laporan medis.
        Jawablah pertanyaan HANYA berdasarkan konteks yang diberikan.

        ATURAN PENTING:
        1. Jawab dengan SINGKAT, PADAT, dan LANGSUNG pada intinya.
        2. DILARANG mengulang-ulang kalimat atau mengulang pertanyaan user.
        3. Jangan menambahkan basa-basi seperti "Berdasarkan konteks yang diberikan...". Langsung berikan jawabannya.
        4. Jika informasi tidak ada di konteks, katakan "Informasi tidak ditemukan".
        <|im_end|>
        <|im_start|>user
        KONTEKS:
        {context}

        PERTANYAAN:
        {question}
        <|im_end|>
        <|im_start|>assistant
        """
        self.prompt = ChatPromptTemplate.from_template(template)

        self.generation_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run(self, query):
        """
        Eksekusi Step-by-Step untuk RAG Fusion
        """
        # 1. Jalankan Fusion Retrieval Chain (Generate Query -> Retrieve -> Rerank)
        # Hasilnya adalah List of Documents yang sudah di-fusi
        fused_docs = self.retrieval_chain_rag_fusion.invoke({"question": query})
        
        # 2. Format Dokumen Fusi ke String
        formatted_context = self.format_docs(fused_docs)
        
        # 3. Generate Answer
        answer = self.generation_chain.invoke({
            "context": formatted_context,
            "question": query
        })
        
        # 4. Return Paket Lengkap
        return {
            "answer": answer,
            "contexts": fused_docs # Ini dokumen hasil fusion (penting untuk evaluasi)
        }
    
if __name__ == "__main__":
    config = {
    'LANGSMITH_ENDPOINT': "https://api.smith.langchain.com",
    'LANGSMITH_PROJECT': "skripsi",
    'LANGSMITH_API_KEY': userdata.get('LANGSMITH_API_KEY'),
    'OPENAI_API_KEY': userdata.get('OPENAI_API_KEY'),
    'QDRANT_API_KEY': userdata.get('QDRANT_API_KEY'),
    'QDRANT_URL': userdata.get('QDRANT_URL'),
    'HF_TOKEN': userdata.get('HF_TOKEN'),
    'EMBEDDING_MODEL': "google/embeddinggemma-300m",
    'COLLECTION_NAME': "medical_reports",
    'MODEL_ID': "Qwen/Qwen3-4B-Instruct-2507"
    }

    print("Menginisialisasi Model dan Vector Database...")
    rag_baseline = RAGBaseline(config)
    rag_fusion = RAGFusion(config)
    print("Inisialisasi selesai!\n")


    input_file = "nutrition_qa_golden_dataset.json"
    output_file = "rag_responses_dataset.jsonl"

    with open(input_file, 'r', encoding='utf-8') as f:
        synthetic_dataset = json.load(f)

    print(f"Memulai eksekusi untuk {len(synthetic_dataset)} pertanyaan...")

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for item in tqdm(synthetic_dataset, desc="Proses Evaluasi RAG"):
            original_question = item['question']
            first_chunk_meta = item['chunk_metadata'][0]
            anthro = first_chunk_meta.get('anthropometric_assessment', '')
            diagnosis = first_chunk_meta.get('medical_diagnosis', '')

            query_to_run = f"Pasien dengan {anthro} dan diagnosis medis {diagnosis}. {original_question}"


            try:
                baseline_res = rag_baseline.run(query_to_run)
                baseline_answer = baseline_res['answer']
                baseline_contexts = [doc.page_content for doc in baseline_res['contexts']]
                baseline_contexts_metadata = [doc.metadata for doc in baseline_res['contexts']]
            except Exception as e:
                print(f"\n[Error Baseline] Gagal memproses pertanyaan: '{original_question[:30]}...'\nError: {e}")
                baseline_answer, baseline_contexts, baseline_ids = "ERROR", [], []

            try:
                fusion_res = rag_fusion.run(query_to_run)
                fusion_answer = fusion_res['answer']
                fusion_contexts = [doc.page_content for doc in fusion_res['contexts']]
                fusion_contexts_metadata = [doc.metadata for doc in fusion_res['contexts']]
            except Exception as e:
                print(f"\n[Error Fusion] Gagal memproses pertanyaan: '{original_question[:30]}...'\nError: {e}")
                fusion_answer, fusion_contexts, fusion_ids = "ERROR", [], []

            result_item = {
                "question": original_question,
                "ground_truth": item['ground_truth'],
                "chunk_metadata": item['chunk_metadata'],

                "baseline_answer": baseline_answer,
                "baseline_retrieved_ids": baseline_ids,
                "baseline_retrieved_contexts": baseline_contexts,

                "fusion_answer": fusion_answer,
                "fusion_retrieved_ids": fusion_ids,
                "fusion_retrieved_contexts": fusion_contexts
            }

            f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
            f_out.flush()

    print("\nEksekusi selesai! Semua data aman tersimpan di disk.")
