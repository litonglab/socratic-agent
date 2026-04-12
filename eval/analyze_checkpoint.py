"""临时脚本：分析当前 checkpoint 数据（排除变体D）"""
import json
import sys
from pathlib import Path

ABLATION_OVERALL_WEIGHTS = {
    "relevance": 0.15,
    "faithfulness": 0.20,
    "completeness": 0.15,
    "technical_accuracy": 0.15,
    "pedagogical_guidance": 0.20,
    "progressive_disclosure": 0.15,
}


def _recalc_overall(row: dict) -> None:
    """从六维原始分重算 overall（覆盖旧整数值）。"""
    vals = [row.get(k) for k in ABLATION_OVERALL_WEIGHTS]
    if any(v is None for v in vals):
        return
    row["overall"] = round(
        sum(v * w for v, w in zip(vals, ABLATION_OVERALL_WEIGHTS.values())), 3
    )


checkpoint = Path(__file__).resolve().parent / "results" / "checkpoints" / "ablation_checkpoint.jsonl"
rows = []
with open(checkpoint, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            _recalc_overall(r)
            rows.append(r)

print(f"总记录数: {len(rows)}")
# 排除变体D + 排除API失败（overall=1且answer很短）
rows = [r for r in rows
        if r["variant"] != "D"
        and not (r.get("overall") == 1 and r.get("answer_length", 0) < 200)]
print(f"有效记录（排除D和API失败）: {len(rows)}")

variants = ["A", "B", "C", "E", "F"]
dims = ["relevance", "faithfulness", "completeness", "technical_accuracy",
        "pedagogical_guidance", "progressive_disclosure", "overall"]
dim_zh = {
    "relevance": "相关性", "faithfulness": "忠实性", "completeness": "完整性",
    "technical_accuracy": "技术准确", "pedagogical_guidance": "引导性",
    "progressive_disclosure": "递进性", "overall": "综合",
}

print()
header = f"  {'变体':<4} {'描述':<26} {'N':>3}"
for d in dims:
    header += f"  {dim_zh[d]:>6}"
print(header)
print("-" * 100)

variant_scores = {}
for v in variants:
    v_rows = [r for r in rows if r["variant"] == v]
    if not v_rows:
        continue
    desc = v_rows[0].get("variant_desc", "")
    scores = {}
    for d in dims:
        vals = [r[d] for r in v_rows if r.get(d) is not None]
        scores[d] = sum(vals) / len(vals) if vals else 0
    variant_scores[v] = scores
    line = f"  {v:<4} {desc:<26} {len(v_rows):>3}"
    for d in dims:
        line += f"  {scores[d]:>6.3f}"
    print(line)

# 模块贡献
if "A" in variant_scores:
    print()
    print("=" * 80)
    print("模块贡献 (Δ = A - 消融变体，正值 = 该模块有正贡献)")
    print("=" * 80)
    a = variant_scores["A"]
    for v in ["B", "C", "E", "F"]:
        if v not in variant_scores:
            continue
        vs = variant_scores[v]
        desc = [r for r in rows if r["variant"] == v][0].get("variant_desc", "")
        print(f"\n  A vs {v} ({desc}):")
        for d in dims:
            delta = a[d] - vs[d]
            marker = "  ✓" if delta > 0 else "  ✗" if delta < 0 else ""
            print(f"    {dim_zh[d]:<8}: Δ = {delta:+.3f}{marker}")

    # A vs C 重点分析
    if "C" in variant_scores:
        print()
        print("=" * 80)
        print("重点：A(完整系统) vs C(-Socratic) 教学维度对比")
        print("=" * 80)
        c = variant_scores["C"]
        for d in dims:
            print(f"  {dim_zh[d]:<8}: A={a[d]:.3f}  C={c[d]:.3f}  Δ={a[d]-c[d]:+.3f}")
