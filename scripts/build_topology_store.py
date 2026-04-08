#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag.topo_rag import build_topology_store


def _resolve_docx_targets(docx_args: list[str], data_dir: str) -> list[Path]:
    if docx_args:
        targets = []
        for raw in docx_args:
            path = Path(raw)
            if not path.is_absolute():
                path = ROOT / raw
            targets.append(path)
        return targets

    base = Path(data_dir)
    if not base.is_absolute():
        base = ROOT / data_dir
    return sorted(base.glob("*.docx"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建拓扑结构化存储：抽图 -> 前置分类 -> GPT-4o 抽取 -> GPT-4o 审核 -> 落盘 approved_json"
    )
    parser.add_argument(
        "--docx",
        nargs="*",
        default=[],
        help="一个或多个 docx 路径；不传则扫描 --data-dir 下的所有 .docx",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="当未显式传入 --docx 时，用于扫描 .docx 的目录，默认 data",
    )
    parser.add_argument(
        "--output-root",
        default="topo_store",
        help="拓扑存储根目录，默认 topo_store",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="手动指定实验 ID（如 lab13）；仅在处理单个 docx 时生效",
    )
    parser.add_argument(
        "--classify-model",
        default="gpt-4o-mini",
        help="用于前置区分 topology/non_topology/unclear 的模型，默认 gpt-4o-mini",
    )
    parser.add_argument(
        "--extract-model",
        default="gpt-4o",
        help="用于初次拓扑抽取的模型，默认 gpt-4o",
    )
    parser.add_argument(
        "--review-model",
        default="gpt-4o",
        help="用于审核与修正拓扑 JSON 的模型，默认 gpt-4o",
    )
    parser.add_argument(
        "--min-image-pixels",
        type=int,
        default=120 * 120,
        help="候选图片最小像素面积，默认 14400",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖重建当前实验目录下的 images/raw_json/reviews/approved_json",
    )
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    targets = _resolve_docx_targets(args.docx, args.data_dir)
    if not targets:
        print("错误：未找到任何 .docx 文件。请传入 --docx 或检查 --data-dir。")
        return 1

    if args.experiment_id and len(targets) != 1:
        parser.error("--experiment-id 仅支持与单个 docx 一起使用。")

    failed = 0
    for idx, docx_path in enumerate(targets, start=1):
        print(f"\n[{idx}/{len(targets)}] 处理：{docx_path}")
        try:
            manifest = build_topology_store(
                str(docx_path),
                output_root=args.output_root,
                experiment_id=args.experiment_id if len(targets) == 1 else None,
                classify_model=args.classify_model,
                extract_model=args.extract_model,
                review_model=args.review_model,
                overwrite=args.overwrite,
                min_image_pixels=args.min_image_pixels,
            )
        except Exception as exc:
            failed += 1
            print(f"  失败：{repr(exc)}")
            continue

        print(
            "  完成："
            f"experiment_id={manifest.experiment_id}, "
            f"images={manifest.images_total}, "
            f"prefiltered_non_topology={manifest.non_topology_prefiltered_total}, "
            f"approved={manifest.approved_total}, "
            f"approved_with_warnings={manifest.approved_with_warnings_total}, "
            f"manual_review={manifest.needs_manual_review_total}, "
            f"rejected={manifest.rejected_non_topology_total}"
        )
        print(f"  输出目录：{manifest.store_dir}")

    if failed:
        print(f"\n完成，但有 {failed} 个文档处理失败。")
        return 1

    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
