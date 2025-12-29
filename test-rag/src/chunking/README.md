# `chunking` 模块说明文档

## 1. 模块概述
`chunking` 模块负责将 `data_loader` 模块解析出的文档元素（文本、表格、图片等）切分成适合索引和检索的、带有丰富元数据的“块”（chunks）。这个过程是 RAG 系统性能的关键，因为它直接影响检索的粒度和上下文的完整性。

## 2. 关键设计
*   **`TextSplitter` 类**: 封装了所有分块逻辑，提供灵活的配置参数。
*   **层次感知切分**: 优先根据文档的标题层级进行切分，确保语义完整性。
*   **固定长度与重叠**: 对大块文本进行固定长度切分，并引入重叠，以保留上下文。
*   **表格处理**: 针对表格内容进行特殊处理，确保其作为独立的或关联的 chunk 被正确索引。
*   **元数据丰富**: 在每个 chunk 中附加详细的元数据，如 `doc_id`, `section_path`, `page_num` 等，用于后续的检索和引用。

## 3. 输入/输出

### 输入
*   `parsed_document_data` (Dict[str, Any]): `data_loader` 模块的 `parse_docx` 方法的输出。包含 `doc_id` 和 `elements` 列表。

### 输出
*   `List[Dict]`: 一个 chunk 字典的列表，每个字典代表一个独立的 chunk。每个 chunk 字典的结构遵循项目前述定义的 `chunk schema`：

```python
{
  "chunk_id": "string",  # 唯一 ID
  "doc_id": "string",    # 原始 docx 文件名
  "section_path": ["H1", "H2", "H3"], # 标题层级路径
  "start_offset": "int", # chunk 在原始文档中的起始字符偏移量
  "end_offset": "int",   # chunk 在原始文档中的结束字符偏移量
  "content_text": "string", # 清理后的文本内容
  "modality": "text | table | figure", # 内容模态
  "page_num": "int",     # 所在页码 (可选)
  "image_path": "string", # 如果是图片描述，指向本地图片路径 (可选)
  "table_html": "string", # 如果是表格，保存为 HTML 格式 (可选)
  "metadata": {          # 其他自定义元数据
    "lab": "e.g., OSPF",
    "topic": "e.g., neighbor",
    "difficulty": "intro|mid|adv",
    "tags": ["troubleshooting", "routing"]
  }
}
```

## 4. 主要类/函数

### `TextSplitter` 类
*   **`__init__(self, chunk_size: int = 512, chunk_overlap: int = 100)`**: 构造函数，初始化分块大小和重叠量。
*   **`split_document(self, parsed_document_data: Dict[str, Any]) -> List[Dict]`**: 核心方法，接收解析后的文档数据并将其切分成 chunk 列表。
    *   **内部辅助方法 (私有)**：
        *   `_get_token_length(text: str) -> int`: 计算文本的 token 长度（使用 `tiktoken` 库）。
        *   `_create_chunk(content: str, element_metadata: Dict) -> Dict`: 创建一个 chunk 字典，并填充 `chunk_id` 和其他元数据。
        *   `_split_text_recursively(text_content: str, current_metadata: Dict) -> List[Dict]`: 对长文本进行固定长度+重叠切分。
        *   `_process_section(elements_in_section: List[Dict]) -> List[Dict]`: 处理一个标题层级下的所有元素。

## 5. 依赖
*   `tiktoken`: 用于精确计算文本的 token 长度，以便控制 chunk 大小。
*   `uuid`: 生成唯一的 `chunk_id`。

## 6. 实现细节

### 分块流程
1.  **初始化 `TextSplitter`**: 配置 `chunk_size` 和 `chunk_overlap`。
2.  **`split_document` 方法:**
    *   接收 `data_loader` 模块的输出 `parsed_document_data`。
    *   遍历 `parsed_document_data["elements"]`，根据 `element["type"]` 和 `element["section_path"]` 来判断如何切分。
    *   **层次感知**: 维护当前的 `section_path`。当遇到不同层级的标题时，意味着一个新的逻辑章节开始，可以作为一个分块的边界。
    *   **处理 "text" 元素:**
        *   将属于同一个 `section_path` 的连续文本内容进行合并。
        *   使用 `_get_token_length` 检查合并后的文本长度。
        *   如果文本过长，则调用 `_split_text_recursively` 进行固定长度 + 重叠切分。
        *   为每个切分后的文本块创建 `type: "text"` 的 chunk。
    *   **处理 "table" 元素:**
        *   将表格内容（Markdown 或 HTML 字符串）作为独立的 chunk。设置 `modality: "table"`，并包含 `table_html` 或 `content_text`。
        *   可以考虑将表格与其上下的文本进行关联，例如将表格的简要描述合并到相邻的文本 chunk 中。
    *   **处理 "image" 元素:**
        *   为图片创建一个独立的 chunk。设置 `modality: "figure"`，`content_text` 可以是图片周围的文本描述（如果 `data_loader` 提取了），或者简要说明“图示”。
        *   包含 `image_path` 和其他图片相关的元数据。
3.  **Chunk ID 生成**: 使用 `uuid.uuid4().hex` 或类似方式为每个生成的 chunk 创建一个唯一的 `chunk_id`。
4.  **元数据填充**: 确保每个 chunk 字典都包含前述 `chunk schema` 中定义的所有必要元数据。

### 中文分词考量
*   `tiktoken` 是面向英文 token 计算的，对于中文会按照字符进行编码，所以 `chunk_size` 需要根据实际情况调整。在 `_get_token_length` 中，可以简单使用 `len(text)` 来统计字符数，或者使用 `jieba` 进行中文分词后统计词数，但通常使用 `tiktoken` 的字节对编码（BPE）来模拟 token 长度更准确。本阶段仅用于长度控制，不进行实际分词。

