# `rag` 模块说明文档

## 1. 模块概述
`rag` 模块负责 RAG（Retrieval-Augmented Generation）系统的最终答案生成。它接收用户查询以及 `retrieval` 模块检索到的相关文档片段（context），然后利用大型语言模型（LLM）来综合这些信息，生成连贯、准确且带有可追溯引用的回答。本模块的核心在于精心设计的提示词工程，以确保 LLM 严格遵循规则，减少幻觉并提供高质量的教学式回答。

## 2. 关键设计
*   **`RAGGenerator` 类**: 封装 LLM 的调用逻辑和提示词管理。
*   **提示词工程**: 使用 `System Prompt` 和 `User Prompt` 模板来指导 LLM 的行为，强制其基于提供上下文回答，并提供结构化的引用。
*   **引用生成**: 负责将检索到的 chunk 的元数据格式化为易于阅读的引用信息，并附加到生成的答案中。
*   **拒答机制**: 通过提示词设计，使 LLM 在上下文不足时能够明确表达“无法回答”。

## 3. 输入/输出

### 输入
*   `query` (str): 原始的用户查询。
*   `context` (str): 由 `retrieval` 模块提供的、经过拼接的相关文档片段文本。
*   `citations` (List[Dict]): 包含每个检索到的 chunk 引用信息的列表，格式为：
    ```python
    [
        {"doc_id": "file.docx", "section_path": ["H1", "H2"], "content_text_summary": "..."},
        # ...
    ]
    ```

### 输出
*   `str`: LLM 生成的最终答案，其中包含格式化的引用信息。

## 4. 主要类/函数

### `RAGGenerator` 类
*   **`__init__(self, llm_client)`**: 构造函数，接收 `models.llm_client` 实例。
*   **`generate_answer(self, query: str, context: str, citations: List[Dict]) -> str`**: 核心方法，根据查询、上下文和引用生成答案。
    *   **内部辅助方法 (私有)**：
        *   `_format_citations(citations: List[Dict]) -> str`: 将引用信息格式化为字符串。
        *   `_build_system_prompt() -> str`: 构建 System Prompt。
        *   `_build_user_prompt(query: str, context: str) -> str`: 构建 User Prompt。

## 5. 依赖
*   `src.models.llm_client`: 用于与 LLM 进行交互（调用 `gpt-4o`）。

## 6. 实现细节

### 生成流程
1.  **初始化 `RAGGenerator`**: 在 `__init__` 中接收 `LLMClient` 实例。
2.  **`generate_answer` 方法:**
    *   **构建提示词**: 调用 `_build_system_prompt` 和 `_build_user_prompt` 方法，将 `query` 和 `context` 嵌入到对应的模板中。
    *   **调用 LLM**: 使用 `self.llm_client.chat_completion(system_prompt, user_prompt)` 调用 `gpt-4o` API。
    *   **处理 LLM 响应**: 提取 LLM 生成的文本内容。
    *   **格式化引用**: 调用 `_format_citations` 方法将 `citations` 列表转换为引用字符串。
    *   **组合答案**: 将 LLM 生成的文本与格式化的引用字符串组合成最终答案。
    *   **返回答案**: 返回包含引用信息的最终答案。

### 提示词内容 (基于项目前述 `F. 生成策略`)
*   **System Prompt**: 强调作为计算机网络实验助教的角色，严格基于上下文回答，并提供准确引用的规则。明确拒答情况的表达。
*   **User Prompt**: 提供用户查询和检索到的上下文。

### 引用格式
```
【引用】
- 文件: [doc_id], 章节: [section_path], 内容: [content_text 摘要 (不超过50字)]
- 文件: [doc_id], 章节: [section_path], 内容: [content_text 摘要 (不超过50字)]
```

### 错误处理/拒答
*   在 `generate_answer` 中，如果 LLM 返回的答案包含“我无法从提供的文档中找到相关信息”的特定短语，并且 `citations` 为空，则可以认为这是一个有效的拒答。否则，需要进一步检查答案的忠实度和相关性（这部分将主要由 `evaluation` 模块处理）。

