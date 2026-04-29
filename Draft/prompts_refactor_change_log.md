# `agentic_rag/prompts.py` 改动说明

本文档记录本次对 [`agentic_rag/prompts.py`](/home/octopus/socratic-agent/agentic_rag/prompts.py) 的保守重构。目标不是重写整套 prompt，而是在**不改现有调用接口**的前提下，解决明显冲突、降低维护成本、补足运行时说明。

---

## 1. 为旧拆分式 Prompt 增加“当前未接线”说明

### 改动位置
- [`agentic_rag/prompts.py:1`](\/home\/octopus\/socratic-agent\/agentic_rag\/prompts.py#L1)

### 改了什么
- 在 `RELEVANCE_PROMPT`、`CATEGORY_DETECT_PROMPT`、`HINT_JUDGE_PROMPT` 上方增加了说明注释。
- 明确这些 prompt 当前不在 `agent.py` 主链路中直接调用，主链路已经改用 `UNIFIED_CLASSIFICATION_PROMPT`。

### 为什么改
- 这 3 个 prompt 仍然留在文件里，但当前主流程已经不走它们。
- 不加说明的话，后续维护者很容易误以为这些常量仍参与运行时分类。

### 预期影响
- 降低误读成本。
- 不影响运行时行为。

---

## 2. 抽取共享身份约束片段 `COMMON_IDENTITY_CONSTRAINT`

### 改动位置
- [`agentic_rag/prompts.py:99` 附近](\/home\/octopus\/socratic-agent\/agentic_rag\/prompts.py)

### 改了什么
- 新增 `COMMON_IDENTITY_CONSTRAINT` 常量。
- 让 `BASE_PROMPT_LAB`、`BASE_PROMPT_THEORY`、`BASE_PROMPT_REVIEW`、`BASE_PROMPT_CALC` 通过字符串拼接复用这一段。

### 为什么改
- 四个主 Prompt 的身份约束完全重复。
- 重复文本会带来两个问题：
  - 修改一次身份规则时，需要同步改 4 处，容易漏。
  - Prompt 维护时更难确认是否完全一致。

### 预期影响
- 运行时输出不应发生语义变化。
- 后续若要调整身份约束，只需改一处。

---

## 3. 抽取共享策略尾部 `COMMON_STRATEGY_FOOTER`

### 改动位置
- [`agentic_rag/prompts.py:107` 附近](\/home\/octopus\/socratic-agent\/agentic_rag\/prompts.py)

### 改了什么
- 新增 `COMMON_STRATEGY_FOOTER` 常量。
- 将 4 个 `BASE_PROMPT_*` 底部关于 `{current_strategy_instruction}` 的重复尾段统一到这里。

### 为什么改
- 这一段在 4 个主 Prompt 中完全重复。
- 它和 `agent.py` 的 [`_prepare_context()`](/home/octopus/socratic-agent/agentic_rag/agent.py#L552) 存在紧耦合，是最不适合分散维护的部分之一。

### 预期影响
- 不改变运行时协议。
- 降低后续修改策略注入规则时的维护风险。

---

## 4. 修复 `BASE_PROMPT_REVIEW` 与 `STRATEGY_REVIEW[2]` 的规则冲突

### 改动位置
- [`agentic_rag/prompts.py:286` 附近](\/home\/octopus\/socratic-agent\/agentic_rag\/prompts.py)
- 关联运行时上限：[`agentic_rag/agent.py:46`](\/home\/octopus\/socratic-agent\/agentic_rag\/agent.py#L46)
- 关联高等级策略：[`agentic_rag/prompts.py:656`](\/home\/octopus\/socratic-agent\/agentic_rag\/prompts.py#L656)

### 改了什么
- 原来主 Prompt 写的是：
  - “严禁直接给出修正后的完整代码（除非 Level 3）”
- 现在改成：
  - “默认先指出错误位置，不直接给出完整修正配置；当当前教学策略明确要求‘完整修正与工程规范’时，允许提供完整配置”

### 为什么改
- 当前系统里 `CONFIG_REVIEW` 的最高等级是 2，不存在 Level 3。
- 但 `STRATEGY_REVIEW[2]` 明确要求给完整修正配置。
- 这会让模型同时收到两条互斥指令，降低一致性。

### 预期影响
- 在高等级配置审查场景里，模型更稳定地给出完整修正方案。
- 在低等级场景里仍保持渐进披露，不会过早直接贴完整配置。

---

## 5. 修复 `BASE_PROMPT_CALC` 与 `STRATEGY_CALC[1]` 的规则冲突

### 改动位置
- [`agentic_rag/prompts.py:359` 附近](\/home\/octopus\/socratic-agent\/agentic_rag\/prompts.py)
- 关联高等级策略：[`agentic_rag/prompts.py:719`](\/home\/octopus\/socratic-agent\/agentic_rag\/prompts.py#L719)

### 改了什么
- 原来主 Prompt 写的是：
  - “严禁直接给出计算结果数字（即使是在 Level 3）”
- 现在改成：
  - “优先展示方法与推导过程；若当前教学策略要求完整演算与结果讲解，可以在完整过程之后给出最终结果，但不能只报结果不讲过程”

### 为什么改
- `STRATEGY_CALC[1]` 明确要求给出完整演算和最终答案。
- 原规则过于绝对，会和高等级策略打架。

### 预期影响
- 模型在计算题高等级场景下更一致：
  - 先讲过程
  - 再给结论
- 同时避免退化成“只报答案”的计算器行为。

---

## 6. 在策略定义处补充运行时上限说明

### 改动位置
- [`agentic_rag/prompts.py:597` 附近](\/home\/octopus\/socratic-agent\/agentic_rag\/prompts.py)
- [`agentic_rag/prompts.py:697` 附近](\/home\/octopus\/socratic-agent\/agentic_rag\/prompts.py)

### 改了什么
- 给 `STRATEGY_REVIEW` 和 `STRATEGY_CALC` 增加注释说明：
  - `CONFIG_REVIEW` 当前最高等级是 2
  - `CALCULATION` 高等级允许给结果，但必须以前置过程为主

### 为什么改
- 运行时上限定义在 `agent.py`，而策略文本定义在 `prompts.py`。
- 如果文件之间没有文字关联，维护者很难第一眼看出哪些等级实际上永远到不了，哪些语义需要和运行时策略对齐。

### 预期影响
- 提升代码可读性。
- 不影响任何运行时逻辑。

---

## 7. 为什么这次没有做更激进的重写

### 没做的事
- 没有删掉旧 Prompt 常量
- 没有把四个 `BASE_PROMPT_*` 改成一个模板函数
- 没有压缩示例、工具说明、`<思考>` 区块格式
- 没有拆分 `UNIFIED_CLASSIFICATION_PROMPT`

### 为什么没做
- 用户要求“小心一点”。
- 这些改动虽然值得做，但已经会影响模型行为分布，不再属于“保守重构”。
- 本次优先解决：
  - 显式矛盾
  - 明显重复
  - 运行时认知错位

### 后续建议
- 下一轮可以考虑把 4 个主 Prompt 重构为：
  - shared core rules
  - category delta
  - strategy delta
  - tool protocol delta
- 那会显著减少 token 和维护成本，但建议单独做、单独验证。

---

## 8. 本次修改后的预期收益总结

### 预期正向效果
- 减少模型接收互斥指令的概率。
- 降低 prompt 维护时“一处改了，另外几处忘了同步”的风险。
- 让 `prompts.py` 和 `agent.py` 的运行时关系更清晰。

### 预期不变的部分
- `agent.py` 的调用接口不变。
- `BASE_PROMPT_*`、`STRATEGY_*`、`UNIFIED_CLASSIFICATION_PROMPT` 等导出名不变。
- `query()` / `query_stream()` 的主流程不变。

### 可能需要观察的点
- 高等级 `CONFIG_REVIEW` 现在更容易稳定输出完整修正配置。
- 高等级 `CALCULATION` 现在更容易在完整演算后给出明确结果。

