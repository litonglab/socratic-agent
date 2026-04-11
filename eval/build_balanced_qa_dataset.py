"""
构建“拓扑题占比可控”的评测集。

默认行为：
1) 从 eval/qa_dataset.json 读取基础题库
2) 自动推断每题是否 requires_topology
3) 合并 eval/topology_question_bank.json 里的拓扑专项题
4) 按指定拓扑占比抽样，输出新数据集

示例：
  python eval/build_balanced_qa_dataset.py
  python eval/build_balanced_qa_dataset.py --topo-ratio 0.4 --target-size 96
  python eval/build_balanced_qa_dataset.py --topo-ratio 0.45 --min-topo-per-experiment 3
  python eval/build_balanced_qa_dataset.py --output eval/qa_dataset_topo_balanced.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "eval" / "qa_dataset.json"
DEFAULT_TOPO_BANK = ROOT / "eval" / "topology_question_bank.json"
DEFAULT_OUTPUT = ROOT / "eval" / "qa_dataset_topo_balanced.json"


_TOPO_STRONG_PATTERNS = [
    r"拓扑",
    r"链路",
    r"经过哪些设备",
    r"最短路径",
    r"连接到哪台设备",
    r"连接到哪个接口",
    r"哪些设备.*互联",
    r"子网.*成员",
    r"分布在",
    r"上行",
    r"接入.*端口",
]
_TOPO_STRONG_RE = re.compile("|".join(_TOPO_STRONG_PATTERNS), re.IGNORECASE)
_TOPO_LINK_RE = re.compile(r"(连接|互联|链路|路径|经过|端口|接口|分布|成员|拓扑)", re.IGNORECASE)
_DEVICE_TOKEN_RE = re.compile(r"(RT\d+|SW\d+|PC[a-zA-Z0-9]+)", re.IGNORECASE)
_EXPERIMENT_RE = re.compile(r"(?:实验\s*|lab[\s_-]?)(\d+)", re.IGNORECASE)


def infer_requires_topology(item: dict) -> bool:
    """启发式判断题目是否需要拓扑信息。"""
    if isinstance(item.get("requires_topology"), bool):
        return bool(item["requires_topology"])

    question = str(item.get("question", ""))
    source = str(item.get("source", ""))
    device_tokens = len(_DEVICE_TOKEN_RE.findall(question))

    score = 0
    if _TOPO_STRONG_RE.search(question):
        score += 3
    if _TOPO_LINK_RE.search(question):
        score += 2
    if device_tokens >= 2:
        score += 1
    if "实验13" in source and device_tokens >= 2 and _TOPO_LINK_RE.search(question):
        score += 1

    return score >= 3


def infer_experiment_id(item: dict) -> str:
    """从显式字段/来源文本推断实验号，统一输出 labX。"""
    explicit = str(item.get("experiment_id", "")).strip()
    if explicit:
        m = _EXPERIMENT_RE.search(explicit)
        if m:
            return f"lab{int(m.group(1))}"
        if explicit.lower().startswith("lab"):
            return explicit.lower()

    merged_text = f"{item.get('source', '')} {item.get('question', '')}"
    m = _EXPERIMENT_RE.search(merged_text)
    if m:
        return f"lab{int(m.group(1))}"
    return "unknown"


def stratified_sample(items: List[dict], n: int, rng: random.Random) -> List[dict]:
    """按 type 分层抽样，题量不足时自动回退为随机补齐。"""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)

    by_type: Dict[str, List[dict]] = {}
    for item in items:
        by_type.setdefault(item.get("type", "unknown"), []).append(item)

    sampled: List[dict] = []
    leftovers: List[dict] = []
    per_type = max(1, n // max(1, len(by_type)))

    for _, bucket in by_type.items():
        bucket_copy = list(bucket)
        rng.shuffle(bucket_copy)
        sampled.extend(bucket_copy[:per_type])
        leftovers.extend(bucket_copy[per_type:])

    # 去重并裁切
    sampled = sampled[:n]
    if len(sampled) < n:
        rng.shuffle(leftovers)
        need = n - len(sampled)
        sampled.extend(leftovers[:need])

    return sampled[:n]


def sample_topology_with_experiment_quota(
    topo_candidates: List[dict],
    topo_take: int,
    min_topo_per_experiment: int,
    rng: random.Random,
) -> List[dict]:
    """拓扑题采样：先满足实验配额，再按类型补齐。"""
    if topo_take <= 0 or not topo_candidates:
        return []

    if min_topo_per_experiment <= 0:
        return stratified_sample(topo_candidates, topo_take, rng)

    grouped: Dict[str, List[dict]] = {}
    for item in topo_candidates:
        exp = item.get("_experiment_id", "unknown")
        if exp == "unknown":
            continue
        grouped.setdefault(exp, []).append(item)

    if not grouped:
        return stratified_sample(topo_candidates, topo_take, rng)

    # 目标配额（可能因 topo_take 不足而降级）
    desired = {exp: min(min_topo_per_experiment, len(items)) for exp, items in grouped.items()}
    desired_total = sum(desired.values())
    if desired_total > topo_take:
        desired = {exp: 0 for exp in grouped.keys()}
        ordered = sorted(grouped.keys())
        while sum(desired.values()) < topo_take:
            progressed = False
            for exp in ordered:
                if desired[exp] < len(grouped[exp]):
                    desired[exp] += 1
                    progressed = True
                    if sum(desired.values()) >= topo_take:
                        break
            if not progressed:
                break

    selected: List[dict] = []
    selected_q = set()
    for exp in sorted(grouped.keys()):
        count = desired.get(exp, 0)
        if count <= 0:
            continue
        picked = stratified_sample(grouped[exp], count, rng)
        for row in picked:
            q = str(row.get("question", "")).strip()
            if q in selected_q:
                continue
            selected_q.add(q)
            selected.append(row)

    remain = topo_take - len(selected)
    if remain > 0:
        leftover = [x for x in topo_candidates if str(x.get("question", "")).strip() not in selected_q]
        selected.extend(stratified_sample(leftover, remain, rng))

    return selected[:topo_take]


def dataset_stats(items: List[dict]) -> dict:
    topo_items = [item for item in items if infer_requires_topology(item)]
    topo_n = len(topo_items)
    topo_by_experiment = Counter(
        infer_experiment_id(item) for item in topo_items
    )
    return {
        "total": len(items),
        "topology_n": topo_n,
        "topology_ratio": round((topo_n / len(items)) if items else 0, 4),
        "type_dist": dict(Counter(item.get("type", "unknown") for item in items)),
        "difficulty_dist": dict(Counter(item.get("difficulty", "unknown") for item in items)),
        "topology_by_experiment": dict(topo_by_experiment),
    }


def _dedupe_by_question(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for item in items:
        q = str(item.get("question", "")).strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(item)
    return out


def _prepare_topology_bank(items: List[dict]) -> List[dict]:
    prepared = []
    for row in items:
        prepared.append(
            {
                "id": row.get("id"),
                "question": row.get("question", ""),
                "type": row.get("type", "concept"),
                "difficulty": row.get("difficulty", "medium"),
                "source": row.get("source", "topology_question_bank"),
                "reference": row.get("reference", ""),
                "experiment_id": row.get("experiment_id", "lab13"),
                "requires_topology": True,
                "_origin": "topology_bank",
            }
        )
    return prepared


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def build_balanced_dataset(
    base_dataset: List[dict],
    topo_bank: List[dict],
    target_size: int,
    topo_ratio: float,
    seed: int,
    min_topo_per_experiment: int = 0,
) -> List[dict]:
    rng = random.Random(seed)

    base_items = [dict(item) for item in base_dataset]
    for item in base_items:
        item["requires_topology"] = infer_requires_topology(item)
        item["_experiment_id"] = infer_experiment_id(item)
        item["_origin"] = "base"

    base_items = _dedupe_by_question(base_items)
    topo_bank_items = _dedupe_by_question(_prepare_topology_bank(topo_bank))
    for item in topo_bank_items:
        item["_experiment_id"] = infer_experiment_id(item)

    base_topo = [x for x in base_items if x["requires_topology"]]
    base_non_topo = [x for x in base_items if not x["requires_topology"]]
    topo_candidates = _dedupe_by_question(base_topo + topo_bank_items)

    if target_size <= 0:
        raise ValueError("target_size 必须 > 0")
    if not (0 <= topo_ratio <= 1):
        raise ValueError("topo_ratio 必须在 [0, 1] 之间")

    target_topo = int(round(target_size * topo_ratio))
    target_non_topo = target_size - target_topo

    topo_take = min(target_topo, len(topo_candidates))
    non_topo_take = min(target_non_topo, len(base_non_topo))

    # 如果一侧不足，尽量由另一侧补齐
    shortfall = target_size - (topo_take + non_topo_take)
    if shortfall > 0:
        extra_topo = max(0, len(topo_candidates) - topo_take)
        fill_from_topo = min(shortfall, extra_topo)
        topo_take += fill_from_topo
        shortfall -= fill_from_topo

    if shortfall > 0:
        extra_non_topo = max(0, len(base_non_topo) - non_topo_take)
        fill_from_non_topo = min(shortfall, extra_non_topo)
        non_topo_take += fill_from_non_topo
        shortfall -= fill_from_non_topo

    topo_sample = sample_topology_with_experiment_quota(
        topo_candidates,
        topo_take=topo_take,
        min_topo_per_experiment=min_topo_per_experiment,
        rng=rng,
    )
    non_topo_sample = stratified_sample(base_non_topo, non_topo_take, rng)

    merged = _dedupe_by_question(topo_sample + non_topo_sample)
    rng.shuffle(merged)
    merged = merged[:target_size]

    # 重排 id，保留原 id 便于追踪
    output: List[dict] = []
    for idx, item in enumerate(merged, 1):
        row = dict(item)
        row["orig_id"] = item.get("id")
        row["id"] = idx
        row.pop("_origin", None)
        row.pop("_experiment_id", None)
        output.append(row)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="构建拓扑占比可控的评测集")
    parser.add_argument("--base", type=str, default=str(DEFAULT_BASE), help="基础题库路径")
    parser.add_argument("--topo-bank", type=str, default=str(DEFAULT_TOPO_BANK), help="拓扑题库路径")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出路径")
    parser.add_argument("--target-size", type=int, default=93, help="输出题目总数")
    parser.add_argument("--topo-ratio", type=float, default=0.4, help="目标拓扑题占比（0~1）")
    parser.add_argument(
        "--min-topo-per-experiment",
        type=int,
        default=0,
        help="每个实验至少抽样的拓扑题数（0 表示不强制）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    base_path = _resolve(args.base)
    topo_bank_path = _resolve(args.topo_bank)
    output_path = _resolve(args.output)

    if not base_path.exists():
        raise FileNotFoundError(f"找不到基础题库：{base_path}")
    if not topo_bank_path.exists():
        raise FileNotFoundError(f"找不到拓扑题库：{topo_bank_path}")

    base_dataset = json.loads(base_path.read_text(encoding="utf-8"))
    topo_bank = json.loads(topo_bank_path.read_text(encoding="utf-8"))

    before = dataset_stats(base_dataset)
    balanced = build_balanced_dataset(
        base_dataset=base_dataset,
        topo_bank=topo_bank,
        target_size=args.target_size,
        topo_ratio=args.topo_ratio,
        seed=args.seed,
        min_topo_per_experiment=max(0, args.min_topo_per_experiment),
    )
    after = dataset_stats(balanced)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(balanced, ensure_ascii=False, indent=2), encoding="utf-8")

    print("已生成平衡数据集：", output_path)
    print(
        f"原始: total={before['total']}, topo={before['topology_n']} "
        f"({before['topology_ratio']:.1%}), type={before['type_dist']}"
    )
    print(f"原始拓扑按实验：{before['topology_by_experiment']}")
    print(
        f"输出: total={after['total']}, topo={after['topology_n']} "
        f"({after['topology_ratio']:.1%}), type={after['type_dist']}"
    )
    print(f"输出拓扑按实验：{after['topology_by_experiment']}")


if __name__ == "__main__":
    main()
