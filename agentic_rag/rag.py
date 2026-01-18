from pydantic import BaseModel
from tqdm import tqdm

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import RetrievalQA
from agentic_rag.llm_config import build_chat_llm

# 加载环境变量，读取本地 .env 文件，里面定义了 OPENAI_API_KEY
#_ = load_dotenv(find_dotenv())

# llm
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = build_chat_llm(temperature=0)

# 1. 加载docx文档
loader = Docx2txtLoader("/Users/baoliliu/Downloads/networking-agent/RAG-Agent/data/实验1-网线配线架与机柜（2025版）.docx")

documents = loader.load()


def from_documents_with_tqdm(documents, embedding, **kwargs):
    """
    最简单的进度条包装
    """
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    print("正在生成向量...")
    vectors = []
    
    # 使用tqdm显示进度
    for text in tqdm(texts, desc="Processing"):
        vector = embedding.embed_documents([text])[0]
        vectors.append(vector)
    
    print("正在构建FAISS索引...")
    
    # 创建FAISS
    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embedding,
        metadatas=metadatas,
        **kwargs
    )

# 使用

# # 2. 创建文本分割器
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,      # 每个chunk的大小
#     chunk_overlap=100,    # chunk之间的重叠部分
#     separators=["\n\n", "\n", "。", "，", " ", ""],  # 分割符
#     length_function=len,
# )
# print("finish loader")
# # 3. 分割文本
# texts = text_splitter.split_documents(documents)
# print(1)
# # 选择向量模型，并灌库

# vectors = []
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# print("正在生成向量...")
# texts=[text.page_content for text in texts]
# # 使用tqdm显示进度
# for text in tqdm(texts, desc="Processing"):
#     #text = text.page_content
#     vector = embeddings.embed_documents([text])[0]
#     vectors.append(vector)
    
# print("正在构建FAISS索引...")
    
#     # 创建FAISS
# db=FAISS.from_embeddings(
#         text_embeddings=list(zip(texts, vectors)),
#         embedding=embeddings
#     )
# db.save_local("faiss_index")  # 会生成两个文件: faiss_index.pkl 和 faiss_index

loaded_vectorstore = FAISS.load_local(
    "faiss_index",  # 保存目录
    embeddings,  # 需要相同的嵌入模型（用于新查询）
    allow_dangerous_deserialization=True  # 新版本需要这个参数
)

# 使用加载的向量库
#db = loaded_vectorstore.similarity_search("查询文本")

print("向量索引已保存到 faiss_index/ 目录")
#db = FAISS.from_documents(texts, OpenAIEmbeddings(model="text-embedding-ada-002"))
# 获取检索器，选择 top-2 相关的检索结果
retriever = loaded_vectorstore.as_retriever(search_kwargs={"k": 2})
print("finish embeding")
# 创建带有 system 消息的模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个对接问题排查机器人。
               你的任务是根据下述给定的已知信息回答用户问题。
               确保你的回复完全依据下述已知信息。不要编造答案。
               请用中文回答用户问题。
               
               已知信息:
               {context} """),
    ("user", "{question}")
])

# 自定义的提示词参数
chain_type_kwargs = {
    "prompt": prompt_template,
}

# 定义RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 使用stuff模式将上下文拼接到提示词中
    chain_type_kwargs=chain_type_kwargs,
    retriever=retriever
)


def RAGAgent(message):
    answer = qa_chain.invoke(message)
    return answer


if __name__ == "__main__":
    context="如何制作网线？"
    answer= RAGAgent(context)
    print(answer)
