# config.py

import os

class Config:
    # 调整 DATA_DIR 路径，使其指向 RAG-Agent 根目录下的 data 文件夹
    DATA_DIR = os.path.join(os.getcwd(), "..", "data") 
    PROCESSED_DATA_DIR = os.path.join(os.getcwd(), "processed_data")
    IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, "images")
    CHUNKS_FILE = os.path.join(PROCESSED_DATA_DIR, "chunks.json")
    INDEX_STORE_DIR = os.path.join(os.getcwd(), "index_store")
    CHROMA_DB_DIR = os.path.join(INDEX_STORE_DIR, "chroma_db")
    BM25_INDEX_FILE = os.path.join(INDEX_STORE_DIR, "bm25_index.pkl")

    # Chunking parameters
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 100

    # Retrieval parameters
    BM25_TOP_K = 5
    VECTOR_TOP_K = 5
    HYBRID_TOP_K = 10 # 混合检索后 rerank 的数量

    # LLM parameters
    LLM_MODEL_NAME = "gpt-4o"
    EMBEDDING_MODEL_NAME = "text-embedding-3-small"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # 确保设置环境变量

    # Evaluation
    QA_DATASET_PATH = os.path.join(os.getcwd(), "evaluation", "qa_dataset.json")

    # Create directories if they don't exist
    @staticmethod
    def setup_directories():
        os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(Config.IMAGES_DIR, exist_ok=True)
        os.makedirs(Config.INDEX_STORE_DIR, exist_ok=True)
        os.makedirs(Config.CHROMA_DB_DIR, exist_ok=True)



