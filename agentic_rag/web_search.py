# agentic_rag/web_search.py
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# 初始化搜索包装器
# region: "wt-wt" 代表无特定区域，或者你可以指定 "cn-zh" 偏向中文结果
wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=3)
search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

def WebSearch(query: str) -> str:
    """
    执行网络搜索并返回摘要结果。
    """
    try:
        # 加上 "computer networking" 后缀可以提高相关性，防止搜到完全无关的内容
        # 或者在 Prompt 中约束 Agent 生成更好的 Query
        print(f"--- [WebSearch] Searching: {query} ---")
        return search_tool.invoke(query)
    except Exception as e:
        return f"网络搜索失败: {str(e)}"