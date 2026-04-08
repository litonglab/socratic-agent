"""
构建多套 FAISS 索引（按 experiment_config.py 中的 INDEX_VARIANTS 定义）

用法：
  python eval/retrieval/build_indices.py                     # 构建所有索引
  python eval/retrieval/build_indices.py --only large_chunk  # 只构建指定方案
  python eval/retrieval/build_indices.py --only large_chunk,enriched

注意：baseline 方案复用已有的 faiss_index/，默认跳过。
      加 --force 可强制重建已有索引。
"""

import os
import re
import sys
import pickle
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from experiment_config import INDEX_VARIANTS, EVAL_CONFIG

DATA_DIR = ROOT / "data"


# ── 文档加载与分块 ──────────────────────────────────────────

def load_and_split(variant_config: dict) -> list:
    """加载所有 docx 并按配置分块，返回 Document 列表。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=variant_config["chunk_size"],
        chunk_overlap=variant_config["chunk_overlap"],
        separators=variant_config.get("separators", ["\n\n", "\n", "。", "，", " ", ""]),
        length_function=len,
        add_start_index=True,
    )

    docx_files = sorted(DATA_DIR.glob("*.docx"))
    if not docx_files:
        raise RuntimeError(f"未在 {DATA_DIR} 找到 .docx 文件")

    all_chunks = []

    for docx_path in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_path))
            docs = loader.load()
        except (zipfile.BadZipFile, Exception) as e:
            print(f"  [跳过] {docx_path.name}: {e}")
            continue

        chunks = splitter.split_documents(docs)
        source_name = docx_path.name

        # 从文件名提取实验信息（用于上下文前缀）
        exp_info = ""
        m = re.match(r"(实验\d+.*?)（", source_name)
        if m:
            exp_info = m.group(1)

        for i, doc in enumerate(chunks):
            # 基础 metadata
            doc.metadata["source"] = source_name
            doc.metadata["chunk_id"] = f"{docx_path.stem}-{i:05d}"

            # 可选：给 chunk 文本加上下文前缀
            if variant_config.get("context_prefix") and exp_info:
                doc.page_content = f"【{exp_info}】\n{doc.page_content}"

        all_chunks.extend(chunks)

    return all_chunks


# ── 构建单套索引 ────────────────────────────────────────────

def build_index(variant_name: str, variant_config: dict, embeddings):
    """构建并保存一套 FAISS 索引 + chunks pickle。"""
    print(f"\n{'='*60}")
    print(f"构建索引：{variant_name}")
    print(f"  chunk_size={variant_config['chunk_size']}, "
          f"overlap={variant_config['chunk_overlap']}, "
          f"context_prefix={variant_config.get('context_prefix', False)}")

    chunks = load_and_split(variant_config)
    print(f"  文档块数：{len(chunks)}")

    index_dir = ROOT / variant_config["index_dir"]
    index_dir.mkdir(parents=True, exist_ok=True)

    # 构建 FAISS 并保存
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_dir))
    print(f"  FAISS 索引已保存：{index_dir}")

    # 保存 chunks pickle（BM25 混合检索需要原始 Document 对象）
    chunks_path = index_dir / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"  Chunks pickle 已保存：{chunks_path}")


# ── 主流程 ──────────────────────────────────────────────────

def main():
    # 解析参数
    force = "--force" in sys.argv
    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            only = set(sys.argv[idx + 1].split(","))

    # 筛选要构建的索引方案
    variants_to_build = {}
    for name, config in INDEX_VARIANTS.items():
        if only is not None and name not in only:
            continue

        index_dir = ROOT / config["index_dir"]
        # baseline 使用现有索引，除非 --force
        if index_dir.exists() and not force:
            print(f"[跳过] {name}：索引已存在 ({index_dir})，加 --force 重建")
            continue

        variants_to_build[name] = config

    if not variants_to_build:
        print("\n没有需要构建的索引。如需重建已有索引，请加 --force 参数。")
        return

    print(f"\n将构建 {len(variants_to_build)} 套索引：{', '.join(variants_to_build.keys())}")

    # 加载 Embedding 模型（所有索引共享）
    embedding_model = EVAL_CONFIG.get("embedding_model", "BAAI/bge-m3")
    print(f"\n加载 Embedding 模型：{embedding_model} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    # 逐个构建
    for name, config in variants_to_build.items():
        build_index(name, config, embeddings)

    print(f"\n{'='*60}")
    print("所有索引构建完成！")
    print("接下来运行：python eval/retrieval/run_experiments.py")


if __name__ == "__main__":
    main()
