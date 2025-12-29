# main.py

import os
import json
from typing import List, Dict, Any

from config import Config
from src.data_loader.document_parser import DocumentParser
from src.chunking.text_splitter import TextSplitter
from src.indexing.indexer import Indexer
from src.retrieval.retriever import Retriever
from src.rag.generator import RAGGenerator
from src.models.embedding_model import EmbeddingModel
from src.models.llm_client import LLMClient

class RAGSystem:
    def __init__(self):
        Config.setup_directories()
        self.document_parser = DocumentParser()
        self.text_splitter = TextSplitter(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        self.embedding_model = EmbeddingModel(Config.EMBEDDING_MODEL_NAME, Config.OPENAI_API_KEY)
        self.llm_client = LLMClient(Config.LLM_MODEL_NAME, Config.OPENAI_API_KEY)
        self.indexer = Indexer(
            Config.CHROMA_DB_DIR, 
            Config.BM25_INDEX_FILE, 
            self.embedding_model
        )
        self.retriever = Retriever(
            self.indexer, 
            bm25_top_k=Config.BM25_TOP_K, 
            vector_top_k=Config.VECTOR_TOP_K, 
            hybrid_top_k=Config.HYBRID_TOP_K
        )
        self.generator = RAGGenerator(
            self.llm_client
        )

    def ingest_documents(self, doc_paths: List[str]):
        """
        解析文档，提取文本、图片、表格，并生成 chunk。
        """
        print(f"开始解析 {len(doc_paths)} 个文档...")
        all_chunks = []
        for doc_path in doc_paths:
            print(f"正在解析文档: {doc_path}")
            parsed_data = self.document_parser.parse_docx(doc_path, Config.IMAGES_DIR)
            chunks = self.text_splitter.split_document(parsed_data)
            all_chunks.extend(chunks)
        
        with open(Config.CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"所有 chunk 已保存到: {Config.CHUNKS_FILE}")
        return all_chunks

    def build_index(self, chunks: List[Dict]):
        """
        根据 chunk 构建 BM25 索引和向量索引。
        """
        print(f"开始构建索引，共 {len(chunks)} 个 chunk...")
        self.indexer.build_indexes(chunks)
        print("索引构建完成。")

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        执行 RAG 查询，返回答案和引用。
        """
        print(f"用户查询: {user_query}")
        retrieved_chunks = self.retriever.retrieve(user_query)
        
        # 将检索到的 chunk 内容拼接成 context
        context = "\n\n".join([chunk["content_text"] for chunk in retrieved_chunks])
        
        # 提取引用信息
        citations = []
        for chunk in retrieved_chunks:
            citations.append({
                "doc_id": chunk["doc_id"],
                "section_path": chunk["section_path"],
                "content_text_summary": chunk["content_text"][:50] + "..." if len(chunk["content_text"]) > 50 else chunk["content_text"]
            })
        
        # 生成答案
        generated_answer = self.generator.generate_answer(user_query, context, citations)
        
        return {
            "answer": generated_answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks # 用于评估
        }

if __name__ == "__main__":
    rag_system = RAGSystem()

    # 1. Ingest Documents
    # 假设 data 目录下有多个 .docx 文件
    data_files = [os.path.join(Config.DATA_DIR, f) for f in os.listdir(Config.DATA_DIR) if f.endswith(".docx")]
    
    # 确保 data 目录下有文档
    if not data_files:
        print(f"请将 Word 文档 (.docx) 放入 '{Config.DATA_DIR}' 目录中。")
    else:
        # 清理旧的 processed_data 和 index_store
        import shutil
        if os.path.exists(Config.PROCESSED_DATA_DIR):
            shutil.rmtree(Config.PROCESSED_DATA_DIR)
        if os.path.exists(Config.INDEX_STORE_DIR):
            shutil.rmtree(Config.INDEX_STORE_DIR)
        Config.setup_directories() # 重新创建目录

        chunks = rag_system.ingest_documents(data_files)
        
        # 2. Build Index
        if chunks:
            rag_system.build_index(chunks)
        
            # 3. Query
            print("\n--- 进行查询 ---")
            queries = [
                "什么是虚拟局域网？",
                "如何配置静态路由？",
                "请介绍一下端口镜像技术。",
                "网络层有哪些协议？",
                "什么是子网划分？",
                "这是一个无关的问题。"
            ]
            
            for query in queries:
                response = rag_system.query(query)
                print(f"\n问题: {query}")
                print(f"回答: {response['answer']}")
                print("引用:")
                for citation in response['citations']:
                    print(f"  - 文件: {citation['doc_id']}, 章节: {citation['section_path']}, 内容: {citation['content_text_summary']}")

