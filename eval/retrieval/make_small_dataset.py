"""
从 qa_dataset.json 抽取小规模评测集（分类型均衡抽样）。

默认策略：
  - theory         -> 原始类型 concept
  - troubleshooting-> 原始类型 troubleshooting
  - config         -> 原始类型 config
  - calc           -> 原始类型 calculation
  - 每类抽 3 题，固定随机种子，保证可复现

用法：
  python eval/retrieval/make_small_dataset.py
  python eval/retrieval/make_small_dataset.py --per-type 3 --seed 42
  python eval/retrieval/make_small_dataset.py --input eval/qa_dataset.json --output eval/qa_dataset_small.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "eval" / "qa_dataset.json"
DEFAULT_OUTPUT = ROOT / "eval" / "qa_dataset_small.json"

# 用户语义类型 -> 数据集真实 type
TYPE_MAPPING: Dict[str, str] = {
    "theory": "concept",
    "troubleshooting": "troubleshooting",
    "config": "config",
    "calc": "calculation",
}


def _qid_int(item: dict) -> int:
    try:
        return int(item.get("id", 0))
    except Exception:
        return 0


def build_small_dataset(
    input_path: Path,
    output_path: Path,
    per_type: int = 3,
    seed: int = 42,
) -> List[dict]:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    grouped: Dict[str, List[dict]] = {k: [] for k in TYPE_MAPPING}
    for item in data:
        raw_type = item.get("type", "")
        for user_type, dataset_type in TYPE_MAPPING.items():
            if raw_type == dataset_type:
                grouped[user_type].append(item)
                break

    rng = random.Random(seed)
    sampled: List[dict] = []

    for user_type in ["theory", "troubleshooting", "config", "calc"]:
        candidates = sorted(grouped[user_type], key=_qid_int)
        if len(candidates) < per_type:
            raise ValueError(
                f"类型 {user_type} 可用题目仅 {len(candidates)}，不足 {per_type} 题"
            )
        picked = rng.sample(candidates, per_type)
        sampled.extend(picked)

    # 按题号升序，便于阅读与断点续跑
    sampled = sorted(sampled, key=_qid_int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    return sampled


def main():
    parser = argparse.ArgumentParser(description="构建小规模分类型评测集")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="输入数据集路径")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出数据集路径")
    parser.add_argument("--per-type", type=int, default=3, help="每类抽样题数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    sampled = build_small_dataset(
        input_path=input_path,
        output_path=output_path,
        per_type=args.per_type,
        seed=args.seed,
    )

    print(f"已生成小数据集：{output_path}")
    print(f"总题数：{len(sampled)}（每类 {args.per_type} 题）")
    print("类型映射：theory=concept, troubleshooting=troubleshooting, config=config, calc=calculation")


if __name__ == "__main__":
    main()
