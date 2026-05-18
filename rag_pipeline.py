import torch
import json
import gc
import os
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.load import dumps, loads
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from config import Config

class RAGPipelineBase:
    """Base class to avoid code duplication between Baseline and Fusion."""
    def __init__(self, config=None):
        self.config = config if config else Config.to_dict()
        self.setup_environment()
        self.setup_llm()
        self.setup_vectorstore()

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
        self.client = QdrantClient(url=self.config['QDRANT_URL'], api_key=self.config['QDRANT_API_KEY'])
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config['EMBEDDING_MODEL'],
            model_kwargs={'device': 'cuda', 'token': self.config['HF_TOKEN']}
        )
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.config['COLLECTION_NAME'],
            embedding=self.embedding_model,
            content_payload_key="page_content", # Ensure consistency with indexing
        )

    @staticmethod
    def format_docs(docs):
        # FIXED: Added Source tagging to help LLM distinguish between patient reports
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            formatted.append(f"--- DOKUMEN {i+1} (Sumber: {source}) ---\n{doc.page_content}")
        return "\n\n".join(formatted)

class RAGBaseline(RAGPipelineBase):
    def __init__(self, config):
        super().__init__(config)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.get('TOP_K', 3)})
        self.setup_generation_chain()

    def setup_generation_chain(self):
        template = """<|im_start|>system
Anda adalah asisten medis profesional. Jawablah pertanyaan klinis HANYA berdasarkan KONTEKS yang diberikan.

ATURAN:
1. Jika konteks berasal dari dokumen yang berbeda (sumber berbeda), jawablah dengan hati-hati dan jangan mencampuradukkan data antar pasien.
2. Jawab langsung pada intinya, singkat, dan padat.
3. Jika informasi tidak ada di konteks, jawab "Informasi tidak ditemukan."
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
        self.generation_chain = self.prompt | self.llm | StrOutputParser()

    def run(self, query):
        docs = self.retriever.invoke(query)
        formatted_context = self.format_docs(docs)
        answer = self.generation_chain.invoke({"context": formatted_context, "question": query})
        
        return {
            "answer": answer,
            "contexts": docs,
            "ids": [doc.metadata.get('chunk_id') for doc in docs] # FIXED: Ensure IDs are returned
        }

class RAGFusion(RAGPipelineBase):
    def __init__(self, config):
        super().__init__(config)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.get('TOP_K', 3) * 2})
        self.top_n = self.config.get('TOP_K', 3)
        self.setup_fusion_pipeline()
        self.setup_generation_chain()

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                # FIXED: Use model_dump_json() for Pydantic v2 compatibility
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
Anda adalah ahli pencarian rekam medis. Pecah pertanyaan menjadi 3 kueri pencarian (keyword) yang spesifik.
Hanya berikan 3 baris teks kueri, tanpa angka atau simbol.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
        prompt = ChatPromptTemplate.from_template(query_gen_template)
        
        # Use OpenAI for query generation as per original code
        self.generate_queries = (
            prompt 
            | ChatOpenAI(model="gpt-4o-mini", temperature=0.2) 
            | StrOutputParser() 
            | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
        )

        self.retrieval_chain = (
            self.generate_queries 
            | self.retriever.map() 
            | self.reciprocal_rank_fusion
        )

    def setup_generation_chain(self):
        template = """<|im_start|>system
Anda adalah asisten medis profesional. Jawablah pertanyaan klinis HANYA berdasarkan KONTEKS yang diberikan.
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
        self.generation_chain = self.prompt | self.llm | StrOutputParser()

    def run(self, query):
        fused_docs = self.retrieval_chain.invoke({"question": query})
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
