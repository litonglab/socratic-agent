# `models` 模块说明文档

## 1. 模块概述
`models` 模块负责封装与外部大型语言模型（LLM）和嵌入模型（Embedding Model）的交互逻辑。它提供了统一的接口来初始化和调用这些模型，使得其他模块（如 `indexing` 和 `rag`）能够以解耦的方式使用它们，而无需关心底层的 API 调用细节。本模块旨在提高代码的可维护性和可替换性。

## 2. 关键设计
*   **`EmbeddingModel` 类**: 封装 `text-embedding-3-small` 嵌入模型的调用逻辑。
*   **`LLMClient` 类**: 封装 `gpt-4o` LLM 的调用逻辑。
*   **统一接口**: 为不同的模型提供简单一致的 `embed` 和 `chat_completion` 接口。
*   **API Key 管理**: 集中处理 API Key 的加载和使用（从环境变量获取）。
*   **错误处理**: 包含对 API 调用失败（如网络错误、认证失败、速率限制）的初步处理。

## 3. 输入/输出

### `EmbeddingModel` 输入/输出
*   **输入**: `texts` (List[str] 或 str): 需要进行嵌入的文本列表或单个文本。
*   **输出**: `List[List[float]]` 或 `List[float]`: 对应输入文本的嵌入向量列表。

### `LLMClient` 输入/输出
*   **输入**: 
    *   `system_prompt` (str): 系统提示词。
    *   `user_prompt` (str): 用户提示词。
*   **输出**: `str`: LLM 生成的文本回答。

## 4. 主要类/函数

### `EmbeddingModel` 类
*   **`__init__(self, model_name: str, api_key: str)`**: 构造函数，初始化 OpenAI 客户端和嵌入模型名称。
*   **`embed_documents(self, texts: List[str]) -> List[List[float]]`**: 对文本列表进行批量嵌入。
*   **`embed_query(self, text: str) -> List[float]`**: 对单个查询文本进行嵌入。

### `LLMClient` 类
*   **`__init__(self, model_name: str, api_key: str)`**: 构造函数，初始化 OpenAI 客户端和 LLM 模型名称。
*   **`chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str`**: 调用 LLM 进行对话补全。

## 5. 依赖
*   `openai`: OpenAI Python 客户端库。
*   `os`: 获取环境变量中的 API Key。

## 6. 实现细节

### `EmbeddingModel` 实现流程
1.  **初始化**: 在 `__init__` 中，使用 `openai.OpenAI(api_key=api_key)` 初始化 OpenAI 客户端。存储 `model_name` (`text-embedding-3-small`)。
2.  **`embed_documents`**: 调用 `client.embeddings.create` 方法，传入 `model` 和 `input` (文本列表)。提取并返回嵌入向量列表。
3.  **`embed_query`**: 类似于 `embed_documents`，但针对单个文本，返回单个嵌入向量。
4.  **错误处理**: 包裹 `try-except` 块，捕获 `openai.APIError` 等异常，进行日志记录或重新抛出自定义异常。

### `LLMClient` 实现流程
1.  **初始化**: 在 `__init__` 中，使用 `openai.OpenAI(api_key=api_key)` 初始化 OpenAI 客户端。存储 `model_name` (`gpt-4o`)。
2.  **`chat_completion`**: 调用 `client.chat.completions.create` 方法，传入 `model`, `messages` 列表（包含 `system_prompt` 和 `user_prompt`），以及 `temperature` 参数。从响应中提取生成的文本内容。
3.  **错误处理**: 包裹 `try-except` 块，捕获 `openai.APIError` 等异常，进行日志记录或重新抛出自定义异常。

### API Key 管理
*   建议从环境变量 `OPENAI_API_KEY` 中加载 API Key。在 `config.py` 中定义获取 API Key 的逻辑，并在初始化 `EmbeddingModel` 和 `LLMClient` 时传入。

