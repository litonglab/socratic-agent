# agentic_rag/web_search.py
from __future__ import annotations

# 延迟导入：DuckDuckGo 库和对象在首次搜索时才创建，避免拖慢 server 启动
_search_tool = None


def _get_search_tool():
    global _search_tool
    if _search_tool is None:
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=3)
        _search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
    return _search_tool


def WebSearch(query: str) -> str:
    """
    执行网络搜索并返回摘要结果。
    """
    try:
        print(f"--- [WebSearch] Searching: {query} ---")
        return _get_search_tool().invoke(query)
    except Exception as e:
        return f"网络搜索失败: {str(e)}"
