# `data_loader` 模块说明文档

## 1. 模块概述
`data_loader` 模块负责从原始的 Word 文档（.docx）中提取结构化和非结构化内容。其主要目标是将非结构化的文档内容（如标题、段落、表格、图片）转换为易于后续处理和索引的中间表示。本模块还将负责图片文件的保存。

## 2. 关键设计
*   **`DocumentParser` 类**: 这是模块的核心，用于封装 Word 文档的解析逻辑。
*   **内容统一表示**: 解析后的文档内容将以一个统一的、包含丰富元数据的 JSON 结构返回，方便 `chunking` 模块使用。
*   **图片处理**: 负责从 Word 文档中提取图片并保存到指定目录，同时在输出的结构中保留图片路径及其在文档中的相对位置信息。
*   **文本清理**: 对提取的文本内容进行初步清理，移除多余空白符等。

## 3. 输入/输出

### 输入
*   `docx_path` (str): 待解析的 Word 文档的完整路径。
*   `images_output_dir` (str): 提取的图片文件保存的目录。

### 输出
*   一个字典，包含以下字段：
    *   `doc_id` (str): 原始 Word 文档的文件名（例如 "实验1.docx"）。
    *   `elements` (List[Dict]): 按顺序排列的文档元素列表。每个元素都是一个字典，包含以下字段：
        *   `type` (str): 元素类型，可以是 "text", "table", "image"。
        *   `content` (str): 如果是 "text"，则为段落文本；如果是 "table"，则为表格的 Markdown 或 HTML 字符串表示；如果是 "image"，则为图片的简要描述或空字符串（因为图片内容暂不OCR）。
        *   `section_path` (List[str]): 当前元素所属的标题层级路径（例如 `["第一章 概述", "1.1 背景"]`）。
        *   `page_num` (int, 可选): 元素所在的页码（如果解析器能获取）。
        *   `start_offset` (int, 可选): 元素在文档文本流中的起始字符偏移量。
        *   `end_offset` (int, 可选): 元素在文档文本流中的结束字符偏移量。
        *   `image_path` (str, 可选): 如果 `type` 是 "image"，则为图片保存的相对路径。
        *   `table_html` (str, 可选): 如果 `type` 是 "table"，则为表格的 HTML 表示。
        *   `raw_element` (Any, 可选): 原始的 `docx` 元素对象，便于调试或高级处理。

### 输出示例
```json
{
  "doc_id": "实验1-网线配线架与机柜.docx",
  "elements": [
    {
      "type": "text",
      "content": "第一章 网线配线架与机柜实验指导",
      "section_path": ["第一章 网线配线架与机柜实验指导"],
      "start_offset": 0,
      "end_offset": 20
    },
    {
      "type": "text",
      "content": "1.1 实验目的",
      "section_path": ["第一章 网线配线架与机柜实验指导", "1.1 实验目的"],
      "start_offset": 21,
      "end_offset": 30
    },
    {
      "type": "image",
      "content": "实验拓扑图",
      "section_path": ["第一章 网线配线架与机柜实验指导", "1.1 实验目的"],
      "image_path": "processed_data/images/实验1-网线配线架与机柜_image_0.png",
      "start_offset": 31,
      "end_offset": 31
    },
    {
      "type": "table",
      "content": "| 列1 | 列2 |
|---|---|
| 数据1 | 数据2 |",
      "section_path": ["第一章 网线配线架与机柜实验指导", "1.1 实验目的"],
      "table_html": "<table>...</table>",
      "start_offset": 32,
      "end_offset": 80
    },
    // ... 更多元素
  ]
}
```

## 4. 主要类/函数

### `DocumentParser` 类
*   **`__init__(self)`**: 构造函数。
*   **`parse_docx(self, docx_path: str, images_output_dir: str) -> Dict[str, Any]`**: 核心方法，负责解析单个 Word 文档。
    *   **内部辅助方法 (私有)**：
        *   `_extract_text_from_paragraph(paragraph)`: 提取段落文本，判断是否为标题，更新 `section_path`。
        *   `_extract_table(table)`: 将 Word 表格转换为 Markdown 或 HTML 字符串。
        *   `_extract_image(image_element, doc_id, images_output_dir, image_index)`: 提取图片，保存到文件，并返回图片路径。
        *   `_clean_text(text)`: 清理提取的文本内容。
        *   `_get_paragraph_offset(paragraph, doc)`: 获取段落在文档中的起始/结束字符偏移量（可能需要遍历整个文档或使用其他库辅助）。

### `file_utils.py`
*   **`save_image(image_bytes: bytes, output_path: str) -> str`**: 将图片二进制数据保存到指定路径，并返回保存路径。
*   **`create_dir_if_not_exists(directory_path: str)`**: 如果目录不存在则创建。

## 5. 依赖
*   `python-docx`: 用于解析 Word 文档。
*   `os`: 文件路径操作。
*   `json` (用于输出示例和内部数据结构)。

## 6. 实现细节

### 文档解析流程
1.  初始化 `DocumentParser` 实例。
2.  调用 `parse_docx` 方法，传入文档路径和图片输出目录。
3.  在 `parse_docx` 内部，使用 `docx.Document(docx_path)` 加载文档。
4.  遍历文档的各个部分（如 `document.paragraphs`, `document.tables`）。
5.  **处理段落:**
    *   根据 `paragraph.style.name` 判断是否为标题（例如 'Heading 1', 'Heading 2'）。
    *   维护一个 `current_section_path` 列表，根据标题级别动态更新。
    *   提取 `paragraph.text`，并进行 `_clean_text` 处理。
    *   将段落信息作为 `type: "text"` 的元素添加到 `elements` 列表。
6.  **处理表格:**
    *   遍历 `document.tables`。
    *   对每个 `table` 调用 `_extract_table` 转换为 Markdown 或 HTML 字符串。
    *   将表格信息作为 `type: "table"` 的元素添加到 `elements` 列表。
7.  **处理图片:**
    *   在 `docx` 中，图片通常嵌入在 `paragraph` 或 `drawing` 元素中。需要遍历 `document.inline_shapes` 或更底层的 XML 结构来查找图片。
    *   提取图片的二进制数据。
    *   为图片生成唯一文件名（`doc_id_image_index.png`），并调用 `file_utils.save_image` 保存。
    *   将图片信息作为 `type: "image"` 的元素添加到 `elements` 列表，包含 `image_path`。
8.  **文本偏移量**: 获取准确的字符偏移量可能比较复杂，如果 `python-docx` 不直接支持，可以先忽略，或考虑后期集成其他更底层的解析库。初期可以设置为 0。
9.  返回解析后的文档结构。
