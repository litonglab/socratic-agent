"""
为 qa_dataset.json 中缺少实验上下文的问题添加实验来源前缀
用法：python eval/add_experiment_context.py

示例：
  原问题："如何在交换机的VLAN接口上配置IP地址？"
  来源：  "实验11-网络层协议分析（2025版）.docx"
  改后：  "在实验11（网络层协议分析）中，如何在交换机的VLAN接口上配置IP地址？"
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QA_DATASET = ROOT / "eval" / "qa_dataset.json"

# 从 source 文件名中提取实验编号和名称
# 例如 "实验11-网络层协议分析（2025版）.docx" → ("实验11", "网络层协议分析")
SOURCE_PATTERN = re.compile(
    r"(实验\d+)\s*[-—]\s*(.+?)(?:（\d{4}版）|$)"
)


def extract_experiment_info(source: str) -> tuple:
    """从 source 文件名提取 (实验编号, 实验名称)"""
    m = SOURCE_PATTERN.search(source.replace(".docx", ""))
    if m:
        return m.group(1), m.group(2).strip()
    return None, None


def question_already_has_context(question: str) -> bool:
    """判断问题文字中是否已经提到了实验编号"""
    return bool(re.search(r"实验\s*\d+", question))


def main():
    if not QA_DATASET.exists():
        print(f"错误：找不到 {QA_DATASET}")
        sys.exit(1)

    with open(QA_DATASET, encoding="utf-8") as f:
        questions = json.load(f)

    modified = 0
    skipped = 0

    for q in questions:
        source = q.get("source", "")
        exp_num, exp_name = extract_experiment_info(source)

        if not exp_num:
            skipped += 1
            continue

        if question_already_has_context(q["question"]):
            skipped += 1
            continue

        # 添加实验上下文前缀
        prefix = f"在{exp_num}（{exp_name}）中，"
        original = q["question"]

        # 如果问题以"在"、"如何"、"请"等开头，直接拼接
        # 如果问题以"如果"、"假设"等条件句开头，用"，"连接
        q["question"] = prefix + original[0].lower() + original[1:]

        modified += 1
        print(f"  [{q['id']}] {original[:30]}... → +{prefix}")

    with open(QA_DATASET, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n完成：修改 {modified} 个问题，跳过 {skipped} 个")
    print(f"已更新：{QA_DATASET}")
    print("\n建议之后重新生成参考答案：python eval/add_references.py --force")


if __name__ == "__main__":
    main()
