"""
tests/test_lab4_routing.py
==========================
实验4专项单元测试（无 LLM 调用，无网络请求）。

测试内容：
  1. 分节检测：_detect_lab4_section 对 20 个正样本和 3 个负样本的准确率
  2. Query 增强：_augment_lab4_query 对 lab4 问题正确追加前缀，保留原问题
  3. 元数据校验：lab4_eval_cases.json 字段完整性与合法性

运行方式：
  cd /path/to/RAG-Agent
  .venv/bin/python -m pytest tests/test_lab4_routing.py -v
"""

import json
import pytest
from pathlib import Path

# ── 被测函数 ────────────────────────────────────────────────────────────────
from agentic_rag.agent import _detect_lab4_section, _augment_lab4_query

# ── 测试数据路径 ─────────────────────────────────────────────────────────────
CASES_PATH = Path(__file__).parent.parent / "test_examples" / "lab4_eval_cases.json"

VALID_SECTIONS = {"4.1_iperf", "4.2_tc", "4.3_mahimahi", "4.4_mininet", "4.6_acceptance"}
VALID_PROBLEM_TYPES = {
    "command_execution",
    "phenomenon_explanation",
    "troubleshooting",
    "report_analysis",
    "screenshot_acceptance",
    "concept_explanation",
}


# ── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def all_cases():
    with open(CASES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


@pytest.fixture(scope="session")
def lab4_cases(all_cases):
    return [c for c in all_cases if c["lab4_expected"]]


@pytest.fixture(scope="session")
def non_lab4_cases(all_cases):
    return [c for c in all_cases if not c["lab4_expected"]]


# ── 1. 元数据校验 ─────────────────────────────────────────────────────────────

class TestMetadata:
    def test_cases_file_exists(self):
        assert CASES_PATH.exists(), f"测试集文件不存在：{CASES_PATH}"

    def test_minimum_case_count(self, all_cases):
        assert len(all_cases) >= 23, f"测试集至少需要 23 条，当前只有 {len(all_cases)} 条"

    def test_at_least_20_positive_cases(self, lab4_cases):
        assert len(lab4_cases) >= 20, f"lab4 正样本至少需要 20 条，当前只有 {len(lab4_cases)} 条"

    def test_at_least_3_negative_cases(self, non_lab4_cases):
        assert len(non_lab4_cases) >= 3, f"负样本至少需要 3 条，当前只有 {len(non_lab4_cases)} 条"

    def test_required_fields_present(self, all_cases):
        required = {"id", "question", "lab4_expected", "expected_behaviors"}
        for c in all_cases:
            missing = required - set(c.keys())
            assert not missing, f"Case [{c.get('id')}] 缺少必填字段：{missing}"

    def test_unique_ids(self, all_cases):
        ids = [c["id"] for c in all_cases]
        duplicates = {i for i in ids if ids.count(i) > 1}
        assert not duplicates, f"存在重复 ID：{duplicates}"

    def test_lab4_cases_have_section_and_type(self, lab4_cases):
        for c in lab4_cases:
            assert c.get("section"), f"Lab4 case [{c['id']}] 缺少 section"
            assert c.get("problem_type"), f"Lab4 case [{c['id']}] 缺少 problem_type"

    def test_sections_are_valid(self, lab4_cases):
        for c in lab4_cases:
            s = c.get("section")
            if s is not None:
                assert s in VALID_SECTIONS, (
                    f"Case [{c['id']}] section 非法：{s}，合法值：{VALID_SECTIONS}"
                )

    def test_problem_types_are_valid(self, all_cases):
        for c in all_cases:
            pt = c.get("problem_type")
            if pt is not None:
                assert pt in VALID_PROBLEM_TYPES, (
                    f"Case [{c['id']}] problem_type 非法：{pt}，合法值：{VALID_PROBLEM_TYPES}"
                )

    def test_expected_behaviors_non_empty(self, lab4_cases):
        for c in lab4_cases:
            behaviors = c.get("expected_behaviors", [])
            assert isinstance(behaviors, list) and len(behaviors) > 0, (
                f"Case [{c['id']}] expected_behaviors 为空"
            )

    def test_section_coverage(self, lab4_cases):
        """每个 section 至少有 2 条测试。"""
        coverage = {}
        for c in lab4_cases:
            s = c.get("section")
            if s:
                coverage[s] = coverage.get(s, 0) + 1
        for section in VALID_SECTIONS:
            count = coverage.get(section, 0)
            assert count >= 2, (
                f"Section [{section}] 只有 {count} 条测试，至少需要 2 条"
            )


# ── 2. 分节检测准确率 ─────────────────────────────────────────────────────────

class TestSectionDetection:
    def test_lab4_positive_questions_detected(self, lab4_cases):
        """lab4=True 的问题必须检测到非 None 的 section。"""
        failures = []
        for c in lab4_cases:
            result = _detect_lab4_section(c["question"])
            if result is None:
                failures.append(f"  未检测到：[{c['id']}] {c['question']}")
        assert not failures, "以下 lab4 问题未被识别:\n" + "\n".join(failures)

    def test_lab4_section_matches_expected(self, lab4_cases):
        """当 section 字段非 None 时，检测结果必须与预期完全匹配。"""
        failures = []
        for c in lab4_cases:
            expected = c.get("section")
            if expected is None:
                continue
            result = _detect_lab4_section(c["question"])
            if result != expected:
                failures.append(
                    f"  [{c['id']}] 期望={expected}, 实际={result}\n"
                    f"    Q: {c['question']}"
                )
        assert not failures, "Section 检测不符:\n" + "\n".join(failures)

    def test_non_lab4_questions_not_detected(self, non_lab4_cases):
        """lab4=False 的问题不能被错误识别为任何 lab4 section。"""
        failures = []
        for c in non_lab4_cases:
            result = _detect_lab4_section(c["question"])
            if result is not None:
                failures.append(
                    f"  [{c['id']}] 被误识别为 {result}\n"
                    f"    Q: {c['question']}"
                )
        assert not failures, "非 lab4 问题被误识别:\n" + "\n".join(failures)

    @pytest.mark.parametrize("question,expected_section", [
        ("iperf3 连接不上服务端怎么办？",                   "4.1_iperf"),
        ("iperf3 UDP 为什么有 jitter 和 loss？",           "4.1_iperf"),
        ("iperf3 -s 测 UDP 时服务端要不要加 -u？",         "4.1_iperf"),
        ("我在 lo 上加了 100ms，为什么 ping 出来是 200ms？", "4.2_tc"),
        ("tc qdisc show dev lo 看到 noqueue 是什么意思？",  "4.2_tc"),
        ("tc netem 是作用在入口还是出口？",                  "4.2_tc"),
        ("ens33 上加规则和 lo 上加规则为什么结果不一样？",   "4.2_tc"),
        ("sudo tc qdisc del dev lo root 报错怎么办？",      "4.2_tc"),
        ("tc qdisc add 报 File exists 怎么办？",            "4.2_tc"),
        ("0.6Mbps 的 Mahimahi trace 文件怎么写？",          "4.3_mahimahi"),
        ("24Mbps 的 trace 文件有什么规律？",                 "4.3_mahimahi"),
        ("mm-delay 30 为什么 RTT 增加 60ms？",              "4.3_mahimahi"),
        ("mm-loss uplink 0.1 mm-loss downlink 0.1 之后 ping 丢包为什么不是刚好 10%？",
                                                            "4.3_mahimahi"),
        ("Mahimahi mm-link 图怎么看？",                     "4.3_mahimahi"),
        ("Mininet 里面怎么让 h1 ping h2？",                "4.4_mininet"),
        ("sudo mn 默认建了什么拓扑？",                       "4.4_mininet"),
        ("pingall 和 h1 ping h2 有什么区别？",              "4.4_mininet"),
        ("如何在 Mininet 中测试某条链路 50ms 延迟和 5% 丢包下的 TCP 带宽？",
                                                            "4.4_mininet"),
        ("实验4报告要交哪些截图？",                          "4.6_acceptance"),
        ("Mininet 退出后网络乱了怎么办？",                   "4.4_mininet"),
        # 负样本
        ("OSPF 邻居起不来怎么办？",                         None),
        ("VLAN trunk 不通怎么办？",                         None),
        ("TCP 三次握手是什么？",                             None),
    ])
    def test_individual_detection(self, question, expected_section):
        result = _detect_lab4_section(question)
        assert result == expected_section, (
            f"期望 section={expected_section}，实际={result}\nQ: {question}"
        )


# ── 3. Query 增强校验 ─────────────────────────────────────────────────────────

class TestQueryAugmentation:
    def test_lab4_queries_augmented(self, lab4_cases):
        """lab4=True 的 query 增强后应与原问题不同。"""
        failures = []
        for c in lab4_cases:
            original = c["question"]
            augmented = _augment_lab4_query(original)
            if augmented == original:
                failures.append(f"  [{c['id']}] 未增强：{original[:60]}")
        assert not failures, "以下 lab4 query 未被增强:\n" + "\n".join(failures)

    def test_original_question_preserved(self, lab4_cases):
        """增强后原问题文本必须完整保留在增强串中。"""
        failures = []
        for c in lab4_cases:
            original = c["question"]
            augmented = _augment_lab4_query(original)
            if original not in augmented:
                failures.append(
                    f"  [{c['id']}] 原问题丢失\n"
                    f"    原：{original}\n"
                    f"    增强：{augmented[:100]}"
                )
        assert not failures, "增强后原问题丢失:\n" + "\n".join(failures)

    def test_lab4_keyword_in_augmented(self, lab4_cases):
        """增强后的 query 必须包含 '实验4' 关键词。"""
        failures = []
        for c in lab4_cases:
            augmented = _augment_lab4_query(c["question"])
            if "实验4" not in augmented:
                failures.append(f"  [{c['id']}] 增强串中缺少'实验4'：{augmented[:80]}")
        assert not failures, "增强串缺少'实验4'关键词:\n" + "\n".join(failures)

    def test_section_prefix_matches_section(self, lab4_cases):
        """分节 prefix 与 _detect_lab4_section 检测结果一致。"""
        from agentic_rag.agent import _LAB4_SECTION_AUGMENTATION, _LAB4_GENERAL_PREFIX

        failures = []
        for c in lab4_cases:
            q = c["question"]
            section = _detect_lab4_section(q)
            augmented = _augment_lab4_query(q)
            expected_prefix = (
                _LAB4_SECTION_AUGMENTATION.get(section, _LAB4_GENERAL_PREFIX)
                if section else _LAB4_GENERAL_PREFIX
            )
            if not augmented.startswith(expected_prefix):
                failures.append(
                    f"  [{c['id']}] prefix 不符\n"
                    f"    期望前缀：{expected_prefix[:60]}\n"
                    f"    实际增强：{augmented[:80]}"
                )
        assert not failures, "增强前缀与 section 不一致:\n" + "\n".join(failures)

    def test_non_lab4_queries_use_general_prefix_only(self):
        """非 lab4 问题直接调用 _augment_lab4_query 时仍追加通用前缀（函数本身不做 lab4 判断）。"""
        from agentic_rag.agent import _LAB4_GENERAL_PREFIX
        q = "OSPF 邻居起不来怎么办？"
        augmented = _augment_lab4_query(q)
        assert augmented.startswith(_LAB4_GENERAL_PREFIX), (
            f"非 lab4 问题增强前缀不符：{augmented[:80]}"
        )
        assert q in augmented, "原问题文本丢失"


# ── 4. 功能常量完整性 ─────────────────────────────────────────────────────────

class TestConstantsIntegrity:
    def test_all_sections_have_augmentation_prefix(self):
        from agentic_rag.agent import _LAB4_SECTION_AUGMENTATION
        for section in VALID_SECTIONS:
            if section == "4.6_acceptance":
                continue  # 可以用 general，跳过
            assert section in _LAB4_SECTION_AUGMENTATION, (
                f"Section [{section}] 缺少 augmentation prefix"
            )

    def test_all_sections_have_keywords(self):
        from agentic_rag.agent import _LAB4_SECTION_KEYWORDS
        for section in VALID_SECTIONS:
            assert section in _LAB4_SECTION_KEYWORDS, (
                f"Section [{section}] 缺少 keyword 列表"
            )
            assert len(_LAB4_SECTION_KEYWORDS[section]) >= 2, (
                f"Section [{section}] keyword 列表少于 2 个"
            )

    def test_augmentation_prefixes_contain_lab4_keyword(self):
        from agentic_rag.agent import _LAB4_SECTION_AUGMENTATION
        for section, prefix in _LAB4_SECTION_AUGMENTATION.items():
            assert "实验4" in prefix, f"Section [{section}] prefix 缺少'实验4'"
