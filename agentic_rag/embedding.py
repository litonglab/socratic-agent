import os
from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from docx import Document as docxdocument

def simple_vectorize_folder(
    folder_path: str,
    save_path: str = "./faiss_index",
    chunk_size: int = 1000
):
    """
    简单版本：向量化文件夹中的所有文档
    
    Args:
        folder_path: 文档文件夹路径
        openai_api_key: OpenAI API密钥
        save_path: 向量库保存路径
        chunk_size: 文本块大小
    """
    # 设置API密钥
    #os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # 初始化组件
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    
    all_documents = []
    
    # 查找所有 .doc 和 .docx 文件
    folder = Path(folder_path)
    doc_files = list(folder.rglob("*.docx")) + list(folder.rglob("*.doc"))
    
    print(f"找到 {len(doc_files)} 个文档文件")
    
    # 处理每个文件
    for file_path in doc_files:
        print(f"处理: {file_path.name}")
        
        try:
            # 加载文档
            if file_path.suffix.lower() == '.docx':
                loader = Docx2txtLoader(str(file_path))
            # else:
            #     from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            #     loader = UnstructuredWordDocumentLoader(str(file_path))
            
            docs = loader.load()
            
            # 添加文件信息到元数据
            for doc in docs:
                doc.metadata["source_file"] = str(file_path)
                doc.metadata["file_name"] = file_path.name
            
            # 分割文本
            split_docs = text_splitter.split_documents(docs)
            all_documents.extend(split_docs)
            
            print(f"  生成 {len(split_docs)} 个文本块")
            
        except Exception as e:
            print(f"  处理失败: {e}")
    
    if not all_documents:
        print("没有找到可处理的文档")
        return None
    
    print(f"\n总共生成 {len(all_documents)} 个文本块")
    print("开始向量化...")
    
    # 创建向量库
    vectorstore = FAISS.from_documents(all_documents, embeddings)
    
    # 保存
    vectorstore.save_local(save_path)
    print(f"向量库已保存到: {save_path}")
    
    return vectorstore

# 快速使用
if __name__ == "__main__":
    # 替换为你的配置
    DOCS_FOLDER = "/Users/baoliliu/Downloads/networking-agent/RAG-Agent/data"
    
    # 一键向量化
    vectorstore = simple_vectorize_folder(
        folder_path=DOCS_FOLDER,
        save_path="./faiss_index"
    )
    
    # 测试查询
    if vectorstore:
        results = vectorstore.similarity_search("查询内容", k=2)
        for i, doc in enumerate(results):
            print(f"\n结果 {i+1}:")
            print(f"  内容: {doc.page_content[:150]}...")
            print(f"  来源: {doc.metadata.get('file_name', '未知')}")