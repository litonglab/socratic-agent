"""
问题数据集生成脚本
用法：python eval/generate_qa_dataset.py

功能：
  使用 GPT-4o 从 data/*.docx 中自动生成高质量测试问题
  输出到 eval/qa_dataset_draft.json，供你人工筛选后保存为 eval/qa_dataset.json

环境变量：
  OPENAI_API_KEY   (必须)
  QA_PER_CHUNK     每块生成几个问题，默认 3
  QA_CHUNK_SIZE    每块字符数，默认 1500
"""

import os
import sys
import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 配置 ──────────────────────────────────────────────
DATA_DIR       = ROOT / "data"
OUTPUT_DRAFT   = ROOT / "eval" / "qa_dataset_draft.json"
VISION_MODEL   = "gpt-4o"
QA_PER_CHUNK   = int(os.getenv("QA_PER_CHUNK", "3"))
CHUNK_SIZE     = int(os.getenv("QA_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP  = 200
# ─────────────────────────────────────────────────────

GENERATE_PROMPT = """\
你是计算机网络课程的出题专家，请根据以下文档片段，生成 {n} 个高质量的测试问题。

要求：
1. 问题必须有明确答案，能从文档中找到依据
2. 问题类型多样，覆盖：概念理解、故障排查、配置步骤、计算题
3. 不要出"根据上文"、"文中提到"之类依赖上下文的问题，问题需独立可读
4. 使用中文，问题要具体，避免过于宽泛

文档片段（来源：{source}）：
{content}

请严格按以下 JSON 数组格式输出，不要有其他文字：
[
  {{
    "question": "问题内容",
    "type": "concept/troubleshooting/config/calculation",
    "difficulty": "easy/medium/hard",
    "source": "{source}"
  }}
]"""


def load_chunks(data_dir: Path) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
        length_function=len,
    )
    chunks = []
    docx_files = sorted(data_dir.glob("*.docx"))
    if not docx_files:
        raise RuntimeError(f"未在 {data_dir} 找到 .docx 文件")

    for docx_path in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_path))
            docs = loader.load()
        except zipfile.BadZipFile:
            print(f"  [跳过] 非法 docx：{docx_path.name}")
            continue
        splits = splitter.split_documents(docs)
        for doc in splits:
            doc.metadata["source"] = docx_path.name
        chunks.extend(splits)

    return chunks


def generate_questions_for_chunk(client: OpenAI, chunk_text: str, source: str) -> list:
    prompt = GENERATE_PROMPT.format(
        n=QA_PER_CHUNK,
        content=chunk_text[:2000],
        source=source,
    )
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=30,
        )
        raw = resp.choices[0].message.content.strip()
        # 去掉可能的 markdown 代码块包裹
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"    [警告] 生成失败：{e}")
        return []


def deduplicate(questions: list, sim_threshold: int = 10) -> list:
    """简单去重：问题文本前 sim_threshold 字相同则视为重复"""
    seen = set()
    result = []
    for q in questions:
        key = q.get("question", "")[:sim_threshold]
        if key not in seen:
            seen.add(key)
            result.append(q)
    return result


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"正在加载文档（来自 {DATA_DIR}）...")
    chunks = load_chunks(DATA_DIR)
    print(f"共 {len(chunks)} 个文档块，每块生成 {QA_PER_CHUNK} 个问题")

    # 抽样：每份文档最多取 5 个块，避免重复和 API 费用过高
    from collections import defaultdict
    per_source: dict = defaultdict(list)
    for c in chunks:
        per_source[c.metadata["source"]].append(c)

    selected_chunks = []
    for src, src_chunks in per_source.items():
        step = max(1, len(src_chunks) // 5)
        selected = src_chunks[::step][:5]
        selected_chunks.extend(selected)

    print(f"抽样后共 {len(selected_chunks)} 个块，预计生成约 {len(selected_chunks) * QA_PER_CHUNK} 个问题\n")

    all_questions = []
    for i, chunk in enumerate(selected_chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        print(f"[{i}/{len(selected_chunks)}] {source} ...", end=" ", flush=True)
        qs = generate_questions_for_chunk(client, chunk.page_content, source)
        print(f"生成 {len(qs)} 个")
        all_questions.extend(qs)
        time.sleep(0.3)  # 避免频率限制

    all_questions = deduplicate(all_questions)
    print(f"\n去重后共 {len(all_questions)} 个问题")

    # 加上 id 字段方便追踪
    for i, q in enumerate(all_questions, 1):
        q["id"] = i

    OUTPUT_DRAFT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DRAFT, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    print(f"\n草稿已保存到：{OUTPUT_DRAFT}")
    print("请打开文件，删除不合适的问题，然后将文件另存为：eval/qa_dataset.json")
    print("之后运行：python eval/evaluate_retrieval.py")


if __name__ == "__main__":
    main()
