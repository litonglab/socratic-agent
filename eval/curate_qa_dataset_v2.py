"""
对现有评测集做“清洗 + 定向补题”，生成质量更稳定的 V2 数据集。

默认流程：
1) 读取当前平衡集（seed）
2) 删除低质量样本（如“文档未覆盖”、参考答案过短）
3) 从候选池（基础题库 + 拓扑题库）补齐到目标题量
4) 维持题型分布与拓扑题占比尽量接近 seed
5) 输出新数据集和清洗报告

示例：
  python eval/curate_qa_dataset_v2.py
  python eval/curate_qa_dataset_v2.py --min-reference-len 24
  python eval/curate_qa_dataset_v2.py --target-size 100 --topo-ratio 0.38
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from build_balanced_qa_dataset import infer_requires_topology


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = ROOT / "eval" / "qa_dataset_topo_balanced.json"
DEFAULT_BASE = ROOT / "eval" / "qa_dataset.json"
DEFAULT_TOPO_BANK = ROOT / "eval" / "topology_question_bank.json"
DEFAULT_OUTPUT = ROOT / "eval" / "qa_dataset_topo_balanced_v2.json"
DEFAULT_REPORT = ROOT / "eval" / "qa_dataset_topo_balanced_v2_report.json"

_TYPE_ORDER = ["concept", "config", "troubleshooting", "calculation", "unknown"]
_TYPO_REPLACEMENTS = {
    "pCb": "PCb",
    "Gagibit": "Gigabit",
}


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _canonical_question(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[，。！？、,.!?：:；;（）()【】\\[\\]\"“”‘’`']", "", value)
    return value


def _apply_typo_fixes(text: str) -> str:
    output = text
    for old, new in _TYPO_REPLACEMENTS.items():
        output = output.replace(old, new)
    return output


def _prepare_topology_bank_items(items: List[dict]) -> List[dict]:
    out = []
    for row in items:
        out.append(
            {
                "id": row.get("id"),
                "orig_id": row.get("id"),
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
    return out


def _normalize_item(item: dict, *, origin: str, fix_typos: bool) -> dict:
    row = dict(item)
    if "orig_id" not in row:
        row["orig_id"] = row.get("id")
    row["_origin"] = origin
    row["question"] = str(row.get("question", "")).strip()
    row["reference"] = str(row.get("reference", "")).strip()
    if fix_typos:
        row["question"] = _apply_typo_fixes(row["question"])
        row["reference"] = _apply_typo_fixes(row["reference"])
    row["requires_topology"] = (
        bool(row["requires_topology"])
        if isinstance(row.get("requires_topology"), bool)
        else infer_requires_topology(row)
    )
    row["type"] = str(row.get("type", "unknown")).strip() or "unknown"
    row["difficulty"] = str(row.get("difficulty", "medium")).strip() or "medium"
    row["source"] = str(row.get("source", "unknown")).strip() or "unknown"
    return row


def _quality_issues(item: dict, min_reference_len: int, drop_uncovered: bool) -> List[str]:
    issues = []
    question = str(item.get("question", "")).strip()
    reference = str(item.get("reference", "")).strip()

    if not question:
        issues.append("empty_question")
    if not reference:
        issues.append("empty_reference")
    if drop_uncovered and "文档未覆盖" in reference:
        issues.append("reference_uncovered")
    if reference and len(reference) < min_reference_len:
        issues.append("reference_too_short")
    return issues


def _dedupe_by_question(items: List[dict]) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for row in items:
        key = _canonical_question(str(row.get("question", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _stats(items: List[dict]) -> dict:
    topo_n = sum(1 for x in items if x.get("requires_topology") is True)
    return {
        "total": len(items),
        "topology_n": topo_n,
        "topology_ratio": round((topo_n / len(items)) if items else 0, 4),
        "type_dist": dict(Counter(x.get("type", "unknown") for x in items)),
        "difficulty_dist": dict(Counter(x.get("difficulty", "unknown") for x in items)),
    }


def _target_type_counts(seed_items: List[dict], target_size: int) -> Dict[str, int]:
    seed_dist = Counter(x.get("type", "unknown") for x in seed_items)
    if not seed_dist:
        return {}

    counts: Dict[str, int] = {}
    running = 0
    types = list(seed_dist.keys())
    for idx, t in enumerate(types):
        if idx == len(types) - 1:
            counts[t] = target_size - running
            break
        ratio = seed_dist[t] / len(seed_items)
        n = int(round(target_size * ratio))
        counts[t] = n
        running += n

    # 由于四舍五入可能出现偏差，做一次兜底调整
    while sum(counts.values()) != target_size and counts:
        diff = target_size - sum(counts.values())
        pivot = max(counts, key=counts.get)
        counts[pivot] += 1 if diff > 0 else -1

    return counts


def _select_next_candidate(
    candidates: List[dict],
    *,
    selected_type_dist: Counter,
    selected_topo_n: int,
    target_type_counts: Dict[str, int],
    target_topo_n: int,
    rng: random.Random,
) -> Tuple[int, dict]:
    best_idx = -1
    best_score = -10**9
    for idx, item in enumerate(candidates):
        score = 0

        t = item.get("type", "unknown")
        cur_t = selected_type_dist.get(t, 0)
        tar_t = target_type_counts.get(t, 0)
        if cur_t < tar_t:
            score += 4
            score += (tar_t - cur_t)

        needs_topo = bool(item.get("requires_topology"))
        topo_deficit = target_topo_n - selected_topo_n
        if topo_deficit > 0 and needs_topo:
            score += 3
        elif topo_deficit < 0 and not needs_topo:
            score += 3
        elif topo_deficit == 0:
            score += 1

        # 轻微优先：优先保留中/高难度，避免过度简单化
        if item.get("difficulty") == "hard":
            score += 1
        elif item.get("difficulty") == "medium":
            score += 0.5

        # 同分随机打散，避免总是选到同一批
        score += rng.random() * 0.01

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        raise RuntimeError("无可选候选题")
    return best_idx, candidates[best_idx]


def _rebalance_topology_ratio(
    selected: List[dict],
    pool: List[dict],
    *,
    target_topo_n: int,
    target_type_counts: Dict[str, int],
    rng: random.Random,
) -> List[dict]:
    """在不显著破坏题型分布的前提下，尽量对齐拓扑题数量。"""
    selected_items = list(selected)
    type_dist = Counter(x.get("type", "unknown") for x in selected_items)
    topo_n = sum(1 for x in selected_items if x.get("requires_topology") is True)

    if topo_n == target_topo_n:
        return selected_items

    def pick_outgoing_index(need_topo: bool) -> int:
        # need_topo=True: 当前拓扑不足，优先踢出非拓扑且题型冗余项
        # need_topo=False: 当前拓扑过多，优先踢出拓扑且题型冗余项
        best_idx = -1
        best_score = -10**9
        for idx, item in enumerate(selected_items):
            is_topo = bool(item.get("requires_topology"))
            if need_topo and is_topo:
                continue
            if (not need_topo) and (not is_topo):
                continue
            t = item.get("type", "unknown")
            surplus = type_dist.get(t, 0) - target_type_counts.get(t, 0)
            score = surplus * 10 + rng.random()
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def pick_incoming(pool_items: List[dict], need_topo: bool, prefer_type: str) -> int:
        best_idx = -1
        best_score = -10**9
        for idx, item in enumerate(pool_items):
            is_topo = bool(item.get("requires_topology"))
            if need_topo and not is_topo:
                continue
            if (not need_topo) and is_topo:
                continue
            score = 0
            if item.get("type", "unknown") == prefer_type:
                score += 5
            score += rng.random()
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    if topo_n < target_topo_n:
        need = target_topo_n - topo_n
        work_pool = list(pool)
        while need > 0:
            out_idx = pick_outgoing_index(need_topo=True)
            if out_idx < 0:
                break
            outgoing = selected_items[out_idx]
            prefer_type = outgoing.get("type", "unknown")
            in_idx = pick_incoming(work_pool, need_topo=True, prefer_type=prefer_type)
            if in_idx < 0:
                break
            incoming = work_pool.pop(in_idx)
            selected_items[out_idx] = incoming
            type_dist[outgoing.get("type", "unknown")] -= 1
            type_dist[incoming.get("type", "unknown")] += 1
            topo_n += 1
            need -= 1

    elif topo_n > target_topo_n:
        need = topo_n - target_topo_n
        work_pool = list(pool)
        while need > 0:
            out_idx = pick_outgoing_index(need_topo=False)
            if out_idx < 0:
                break
            outgoing = selected_items[out_idx]
            prefer_type = outgoing.get("type", "unknown")
            in_idx = pick_incoming(work_pool, need_topo=False, prefer_type=prefer_type)
            if in_idx < 0:
                break
            incoming = work_pool.pop(in_idx)
            selected_items[out_idx] = incoming
            type_dist[outgoing.get("type", "unknown")] -= 1
            type_dist[incoming.get("type", "unknown")] += 1
            topo_n -= 1
            need -= 1

    return selected_items


def curate_dataset(
    *,
    seed_items: List[dict],
    candidate_items: List[dict],
    target_size: int,
    target_topo_ratio: float,
    min_reference_len: int,
    drop_uncovered: bool,
    seed: int,
) -> Tuple[List[dict], dict]:
    rng = random.Random(seed)

    # 1) 清洗 seed
    cleaned_seed: List[dict] = []
    removed_records: List[dict] = []
    for row in _dedupe_by_question(seed_items):
        issues = _quality_issues(row, min_reference_len=min_reference_len, drop_uncovered=drop_uncovered)
        if issues:
            removed_records.append(
                {
                    "id": row.get("id"),
                    "orig_id": row.get("orig_id"),
                    "source": row.get("source"),
                    "question": row.get("question"),
                    "issues": issues,
                }
            )
            continue
        cleaned_seed.append(row)

    # 2) 准备候选池
    selected_keys = {_canonical_question(x.get("question", "")) for x in cleaned_seed}
    pool: List[dict] = []
    for row in _dedupe_by_question(candidate_items):
        key = _canonical_question(row.get("question", ""))
        if not key or key in selected_keys:
            continue
        issues = _quality_issues(row, min_reference_len=min_reference_len, drop_uncovered=drop_uncovered)
        if issues:
            continue
        pool.append(row)

    # 3) 目标分布
    target_topo_n = int(round(target_size * target_topo_ratio))
    target_type_counts = _target_type_counts(seed_items, target_size)

    selected = list(cleaned_seed)
    selected_type_dist = Counter(x.get("type", "unknown") for x in selected)
    selected_topo_n = sum(1 for x in selected if x.get("requires_topology") is True)

    # 4) 补齐
    fill_log: List[dict] = []
    while len(selected) < target_size and pool:
        idx, picked = _select_next_candidate(
            pool,
            selected_type_dist=selected_type_dist,
            selected_topo_n=selected_topo_n,
            target_type_counts=target_type_counts,
            target_topo_n=target_topo_n,
            rng=rng,
        )
        pool.pop(idx)
        selected.append(picked)
        selected_type_dist[picked.get("type", "unknown")] += 1
        if picked.get("requires_topology") is True:
            selected_topo_n += 1
        fill_log.append(
            {
                "orig_id": picked.get("orig_id"),
                "source": picked.get("source"),
                "type": picked.get("type"),
                "requires_topology": picked.get("requires_topology"),
            }
        )

    # 4.5) 二次调平：尽量对齐拓扑题占比
    selected = _rebalance_topology_ratio(
        selected,
        pool,
        target_topo_n=target_topo_n,
        target_type_counts=target_type_counts,
        rng=rng,
    )

    # 5) 如仍不足，返回当前可得最大集合
    if len(selected) < target_size:
        print(f"[警告] 候选池不足，目标 {target_size} 题，实际仅 {len(selected)} 题。")

    rng.shuffle(selected)
    selected = selected[:target_size]

    # 6) 重排 id
    output: List[dict] = []
    for i, row in enumerate(selected, 1):
        item = dict(row)
        item["id"] = i
        item.pop("_origin", None)
        output.append(item)

    report = {
        "target_size": target_size,
        "target_topology_ratio": target_topo_ratio,
        "min_reference_len": min_reference_len,
        "drop_uncovered": drop_uncovered,
        "removed_count": len(removed_records),
        "removed_items": removed_records,
        "filled_count": len(fill_log),
        "filled_items_sample": fill_log[:30],
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗并重建高质量 V2 评测集")
    parser.add_argument("--seed-dataset", type=str, default=str(DEFAULT_SEED), help="当前数据集路径")
    parser.add_argument("--base-dataset", type=str, default=str(DEFAULT_BASE), help="基础题库路径")
    parser.add_argument("--topo-bank", type=str, default=str(DEFAULT_TOPO_BANK), help="拓扑题库路径")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="输出数据集路径")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT), help="清洗报告路径")
    parser.add_argument("--target-size", type=int, default=0, help="目标题数，默认沿用 seed 题数")
    parser.add_argument("--topo-ratio", type=float, default=-1.0, help="目标拓扑题占比，默认沿用 seed 占比")
    parser.add_argument("--min-reference-len", type=int, default=20, help="参考答案最小长度")
    parser.add_argument("--drop-uncovered", dest="drop_uncovered", action="store_true", help="删除“文档未覆盖”题目")
    parser.add_argument("--keep-uncovered", dest="drop_uncovered", action="store_false", help="保留“文档未覆盖”题目")
    parser.add_argument("--fix-typos", dest="fix_typos", action="store_true", help="自动修正已知拼写噪声")
    parser.add_argument("--no-fix-typos", dest="fix_typos", action="store_false", help="不做拼写修正")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.set_defaults(drop_uncovered=True, fix_typos=True)
    args = parser.parse_args()

    seed_path = _resolve(args.seed_dataset)
    base_path = _resolve(args.base_dataset)
    topo_bank_path = _resolve(args.topo_bank)
    output_path = _resolve(args.output)
    report_path = _resolve(args.report)

    for path in [seed_path, base_path, topo_bank_path]:
        if not path.exists():
            raise FileNotFoundError(f"找不到输入文件：{path}")

    drop_uncovered = bool(args.drop_uncovered)
    fix_typos = bool(args.fix_typos)

    seed_raw = json.loads(seed_path.read_text(encoding="utf-8"))
    base_raw = json.loads(base_path.read_text(encoding="utf-8"))
    topo_raw = json.loads(topo_bank_path.read_text(encoding="utf-8"))

    seed_items = [_normalize_item(x, origin="seed", fix_typos=fix_typos) for x in seed_raw]
    base_items = [_normalize_item(x, origin="base", fix_typos=fix_typos) for x in base_raw]
    topo_items = [_normalize_item(x, origin="topology_bank", fix_typos=fix_typos) for x in _prepare_topology_bank_items(topo_raw)]

    target_size = args.target_size if args.target_size > 0 else len(seed_items)
    seed_stats = _stats(seed_items)
    target_topo_ratio = args.topo_ratio if args.topo_ratio >= 0 else seed_stats["topology_ratio"]
    if not (0 <= target_topo_ratio <= 1):
        raise ValueError("topo-ratio 必须在 [0,1] 内")

    candidate_items = _dedupe_by_question(base_items + topo_items + seed_items)

    curated, report = curate_dataset(
        seed_items=seed_items,
        candidate_items=candidate_items,
        target_size=target_size,
        target_topo_ratio=target_topo_ratio,
        min_reference_len=args.min_reference_len,
        drop_uncovered=drop_uncovered,
        seed=args.seed,
    )

    before = _stats(seed_items)
    after = _stats(curated)
    report.update(
        {
            "input_dataset": str(seed_path),
            "base_dataset": str(base_path),
            "topology_bank": str(topo_bank_path),
            "before": before,
            "after": after,
            "high_quality_candidate_topology_n": sum(
                1
                for item in candidate_items
                if not _quality_issues(item, min_reference_len=args.min_reference_len, drop_uncovered=drop_uncovered)
                and item.get("requires_topology") is True
            ),
            "high_quality_candidate_total_n": sum(
                1
                for item in candidate_items
                if not _quality_issues(item, min_reference_len=args.min_reference_len, drop_uncovered=drop_uncovered)
            ),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(curated, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("已生成 V2 数据集：", output_path)
    print(
        f"清洗前: total={before['total']}, topo={before['topology_n']} "
        f"({before['topology_ratio']:.1%}), type={before['type_dist']}"
    )
    print(
        f"清洗后: total={after['total']}, topo={after['topology_n']} "
        f"({after['topology_ratio']:.1%}), type={after['type_dist']}"
    )
    print(f"删除低质量题：{report['removed_count']}，补入题目：{report['filled_count']}")
    print("清洗报告：", report_path)


if __name__ == "__main__":
    main()
