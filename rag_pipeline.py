import torch
import json
import gc
import os
import pickle
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.load import dumps, loads
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from config.config import Config

class RAGPipelineBase:
    """Base class to avoid code duplication between Baseline and Fusion."""
    
    # Standardized generation prompt for fair comparison
    GEN_PROMPT_TEMPLATE = """<|im_start|>system
Anda adalah asisten medis profesional. Jawablah pertanyaan klinis berdasarkan KONTEKS yang diberikan.

ATURAN:
1. Jika konteks berasal dari dokumen yang berbeda (sumber berbeda), jawablah dengan hati-hati dan jangan mencampuradukkan data antar pasien.
2. Jawablah secara detail namun tetap pada intinya, pastikan semua poin teknis medis yang relevan dari konteks disebutkan.
3. Jika informasi pendukung tidak lengkap di konteks, jawablah sesuai informasi yang ada tanpa mengarang fakta medis tambahan. Jika benar-benar tidak ada informasi, nyatakan "Informasi tidak lengkap."
<|im_end|>
<|im_start|>user
KONTEKS:
{context}

PERTANYAAN:
{question}
<|im_end|>
<|im_start|>assistant
"""

    def __init__(self, config=None, llm=None, vectorstore=None):
        self.config = config if config else Config.to_dict()
        self.setup_environment()
        
        if llm:
            self.llm = llm
        else:
            self.setup_llm()
            
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            self.setup_vectorstore()
        
        self.setup_bm25()

    def setup_environment(self):
        # Set environment variables for LangChain and others
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = self.config['LANGSMITH_ENDPOINT']
        os.environ["LANGCHAIN_PROJECT"] = self.config['LANGSMITH_PROJECT']
        os.environ["LANGCHAIN_API_KEY"] = self.config['LANGSMITH_API_KEY']
        os.environ["OPENAI_API_KEY"] = self.config['OPENAI_API_KEY']
        os.environ["QDRANT_API_KEY"] = self.config['QDRANT_API_KEY']
        os.environ["QDRANT_URL"] = self.config['QDRANT_URL']
        os.environ["HF_TOKEN"] = self.config['HF_TOKEN']

    def setup_llm(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config['MODEL_ID'])
        model = AutoModelForCausalLM.from_pretrained(
            self.config['MODEL_ID'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # FIXED: Added stop_sequences and explicit ChatML handling
        text_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            repetition_penalty=1.1,
            return_full_text=False, # FIXED: Don't return the prompt
            stop_sequence=["<|im_end|>", "<|endoftext|>"]
        )
        self.llm = HuggingFacePipeline(pipeline=text_pipe)

    def setup_vectorstore(self):
        self.client = QdrantClient(url=self.config['QDRANT_URL'], api_key=self.config['QDRANT_API_KEY'], timeout=300)
        
        # Ensure payload index exists for metadata filtering
        try:
            from qdrant_client.http import models
            self.client.create_payload_index(
                collection_name=self.config['COLLECTION_NAME'],
                field_name="metadata.medical_diagnosis",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True
            )
        except Exception as e:
            print(f"Index creation skipped/failed: {e}")
            
        # Quantization config for embedding model
        embedding_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config['EMBEDDING_MODEL'],
            model_kwargs={
                'device': 'cuda', 
                'token': self.config['HF_TOKEN'],
                'trust_remote_code': True,
                'quantization_config': embedding_bnb_config,
                'torch_dtype': torch.float16,
            }
        )
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.config['COLLECTION_NAME'],
            embedding=self.embedding_model,
            content_payload_key="page_content", # Ensure consistency with indexing
        )

    def setup_bm25(self):
        """Initializes BM25 for sparse retrieval."""
        pickle_path = self.config.get('PICKLE_PATH', 'knowledge-base/medical_documents_enriched.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                docs = pickle.load(f)
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = self.config.get('TOP_K', 3)
        else:
            print(f"Warning: Pickle file not found at {pickle_path}. BM25 disabled.")
            self.bm25_retriever = None

    @staticmethod
    def format_docs(docs):
        # FIXED: Added Source tagging to help LLM distinguish between patient reports
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            formatted.append(f"--- DOKUMEN {i+1} (Sumber: {source}) ---\n{doc.page_content}")
        return "\n\n".join(formatted)

    def hybrid_search(self, query, k=3, filter_dict=None):
        """Combines Vector Search and BM25."""
        # 1. Vector Search
        instruction = "Representasikan pertanyaan medis ini untuk pencarian rekam medis yang relevan: "
        if filter_dict:
            from qdrant_client.http import models as rest
            must_conditions = [
                rest.FieldCondition(key=f"metadata.{k}", match=rest.MatchValue(value=v))
                for k, v in filter_dict.items() if v
            ]
            qdrant_filter = rest.Filter(must=must_conditions) if must_conditions else None
            vector_docs = self.vectorstore.similarity_search(instruction + query, k=k, filter=qdrant_filter)
        else:
            vector_docs = self.vectorstore.similarity_search(instruction + query, k=k)

        # 2. BM25 Search
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            if filter_dict:
                # Manual filtering for BM25 since it doesn't support metadata filters natively
                bm25_docs = [
                    doc for doc in bm25_docs 
                    if all(doc.metadata.get(mk) == mv for mk, mv in filter_dict.items())
                ]
            bm25_docs = bm25_docs[:k]
        else:
            bm25_docs = []

        # 3. Ensemble (Deduplicate)
        combined = {doc.page_content: doc for doc in vector_docs + bm25_docs}
        return list(combined.values())[:k]

class RAGBaseline(RAGPipelineBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.setup_generation_chain()

    def setup_generation_chain(self):
        self.prompt = ChatPromptTemplate.from_template(self.GEN_PROMPT_TEMPLATE)
        self.generation_chain = self.prompt | self.llm | StrOutputParser()

    def run(self, query, filter_dict=None):
        docs = self.hybrid_search(query, k=self.config.get('TOP_K', 3), filter_dict=filter_dict)
        formatted_context = self.format_docs(docs)
        answer = self.generation_chain.invoke({"context": formatted_context, "question": query})
        
        return {
            "answer": answer,
            "contexts": docs,
            "ids": [doc.metadata.get('chunk_id') for doc in docs]
        }

class RAGFusion(RAGPipelineBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.top_n = self.config.get('TOP_K', 3)
        self.setup_fusion_pipeline()
        self.setup_generation_chain()

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                try:
                    doc_str = doc.model_dump_json()
                except:
                    doc_str = json.dumps({"page_content": doc.page_content, "metadata": doc.metadata})
                
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
        
        reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_docs = []
        for doc_json, _ in reranked[:self.top_n]:
            data = json.loads(doc_json)
            final_docs.append(Document(page_content=data['page_content'], metadata=data['metadata']))
        return final_docs

    def setup_fusion_pipeline(self):
        query_gen_template = """<|im_start|>system
Anda adalah ahli pencarian rekam medis. Tugas Anda:
1. Bedah pertanyaan pengguna menjadi sub-pertanyaan yang lebih sederhana jika kompleks.
2. Buat variasi istilah medis (misal: "BBLR" menjadi "Berat Badan Lahir Rendah").
3. JANGAN hilangkan informasi penting pengidentifikasi pasien (diagnosis, usia, gender).
Berikan 3 variasi pertanyaan, satu per baris, tanpa angka.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
        prompt = ChatPromptTemplate.from_template(query_gen_template)
        self.query_generator = (
            prompt 
            | ChatOpenAI(model="gpt-4o-mini", temperature=0.2) 
            | StrOutputParser() 
            | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
        )

    def setup_generation_chain(self):
        self.prompt = ChatPromptTemplate.from_template(self.GEN_PROMPT_TEMPLATE)
        self.generation_chain = self.prompt | self.llm | StrOutputParser()

    def run(self, query, filter_dict=None):
        generated_queries = self.query_generator.invoke({"question": query})
        all_queries = [query] + generated_queries
        
        all_results = []
        for q in all_queries:
            # Multi-query hybrid search
            docs = self.hybrid_search(q, k=self.top_n * 2, filter_dict=filter_dict)
            all_results.append(docs)
                
        fused_docs = self.reciprocal_rank_fusion(all_results)
        formatted_context = self.format_docs(fused_docs)
        answer = self.generation_chain.invoke({
            "context": formatted_context,
            "question": query
        })
        
        return {
            "answer": answer,
            "contexts": fused_docs,
            "ids": [doc.metadata.get('chunk_id') for doc in fused_docs]
        }
