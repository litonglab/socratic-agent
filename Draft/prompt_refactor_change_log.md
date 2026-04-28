# Prompt 改动说明

文件：
- [agentic_rag/prompts.py](/home/octopus/socratic-agent/agentic_rag/prompts.py)

目标：
- 在不改现有导出名和主调用接口的前提下，做一次保守重构。
- 优先消除规则冲突，减少高重复文本，降低后续维护成本。
- 不主动改变 `agent.py` 当前的拼装方式与运行时路由逻辑。

## 改动 1：为旧版拆分式 Prompt 增加状态说明

位置：
- [agentic_rag/prompts.py:1](/home/octopus/socratic-agent/agentic_rag/prompts.py#L1)

改了什么：
- 给 `RELEVANCE_PROMPT`
- `CATEGORY_DETECT_PROMPT`
- `HINT_JUDGE_PROMPT`

前面补充了说明，明确这些 Prompt 目前是“保留但未直接接入主链路”的旧版资产。

原因：
- 当前主链路实际使用的是 `UNIFIED_CLASSIFICATION_PROMPT`。
- 原文件里旧 Prompt 和新 Prompt 并存，但没有说明，容易让维护者误判哪些在生效。

预期：
- 降低误读成本。
- 后续如果继续清理旧 Prompt，可以更明确地区分“运行中”和“历史保留”。

## 改动 2：抽取共享身份约束常量 `COMMON_IDENTITY_CONSTRAINT`

位置：
- [agentic_rag/prompts.py:99](/home/octopus/socratic-agent/agentic_rag/prompts.py#L99)

改了什么：
- 新增 `COMMON_IDENTITY_CONSTRAINT`
- 把 4 个 `BASE_PROMPT_*` 里重复的“身份约束”替换为共享片段拼接

原因：
- 4 个主 Prompt 中这段文字几乎完全重复。
- 重复文本会造成维护漂移：后面如果改“身份表述”或“禁止透露底模”规则，容易漏改。

预期：
- 后续修改身份规则时只改一处。
- 降低不同主 Prompt 之间出现不一致的风险。

## 改动 3：抽取共享策略尾部常量 `COMMON_STRATEGY_FOOTER`

位置：
- [agentic_rag/prompts.py:105](/home/octopus/socratic-agent/agentic_rag/prompts.py#L105)

改了什么：
- 新增 `COMMON_STRATEGY_FOOTER`
- 把 4 个 `BASE_PROMPT_*` 尾部重复的“当前教学策略与约束”块统一为共享片段

原因：
- 这段文字是高重复块，且是每个主 Prompt 的关键约束之一。
- 如果未来需要调整 hint level 的统一说明，原先要改 4 次。

预期：
- 降低重复修改成本。
- 保证 4 个主 Prompt 在 hint 约束说明上保持一致。

## 改动 4：修正 `BASE_PROMPT_REVIEW` 与 `STRATEGY_REVIEW` 的等级冲突

位置：
- [agentic_rag/prompts.py:287](/home/octopus/socratic-agent/agentic_rag/prompts.py#L287)
- [agentic_rag/prompts.py:597](/home/octopus/socratic-agent/agentic_rag/prompts.py#L597)
- [agentic_rag/agent.py:46](/home/octopus/socratic-agent/agentic_rag/agent.py#L46)

改了什么：
- `BASE_PROMPT_REVIEW` 中原来的规则是“严禁直接给出修正后的完整代码（除非 Level 3）”
- 修改为“默认渐进披露；当当前教学策略明确要求完整修正时，允许提供完整配置”

原因：
- 当前运行时里 `CONFIG_REVIEW` 的最高等级是 2，不存在 Level 3。
- 但 `STRATEGY_REVIEW[2]` 明确要求提供完整修正配置。
- 原表述会让模型同时收到两条互相冲突的指令。

预期：
- 让主 Prompt 与策略 Prompt 的行为一致。
- 降低配置审查类回答“该不该给完整配置”的摇摆。

## 改动 5：修正 `BASE_PROMPT_CALC` 与 `STRATEGY_CALC` 的规则冲突

位置：
- [agentic_rag/prompts.py:359](/home/octopus/socratic-agent/agentic_rag/prompts.py#L359)
- [agentic_rag/prompts.py:697](/home/octopus/socratic-agent/agentic_rag/prompts.py#L697)

改了什么：
- 原来 `BASE_PROMPT_CALC` 明确写“严禁直接给出计算结果数字（即使是在 Level 3）”
- 现在改成：
  - 优先展示方法和推导过程
  - 若高等级策略要求完整演算，可以在完整过程之后给最终结果
  - 但不能只报结果不讲过程

原因：
- `STRATEGY_CALC[1]` 明确要求“完整演算与结果讲解”，包含最终答案。
- 原规则会让模型在计算类问题上收到互相打架的高层指令。

预期：
- 保住“过程优先”的教学目标。
- 同时允许高等级下稳定输出完整解答，减少模型自相矛盾。

## 改动 6：补充 `STRATEGY_REVIEW` 和 `STRATEGY_CALC` 的运行时说明注释

位置：
- [agentic_rag/prompts.py:597](/home/octopus/socratic-agent/agentic_rag/prompts.py#L597)
- [agentic_rag/prompts.py:697](/home/octopus/socratic-agent/agentic_rag/prompts.py#L697)

改了什么：
- 在策略字典前面增加注释，明确：
  - `CONFIG_REVIEW` 的当前最高等级是 2
  - `CALCULATION` 的高等级允许“给结果，但必须连同过程”

原因：
- 这两点是当前行为里最容易被误解的地方。
- 逻辑其实散落在 `agent.py` 的 `_MAX_HINT_LEVEL` 和 `prompts.py` 的文本里，不够直观。

预期：
- 方便后续维护者从 `prompts.py` 单文件就理解行为边界。

## 没有改动的部分

以下内容本次故意没有动：
- `UNIFIED_CLASSIFICATION_PROMPT` 的结构和内容
- 4 组 `STRATEGY_*` 的主体教学逻辑
- `agent.py` 中的 `_BASE_PROMPTS`、`_STRATEGIES`、`_prepare_context()` 拼接方式
- 旧 Prompt 的删除与迁移

原因：
- 这些改动风险更高，会直接影响分类稳定性或运行时输出风格。
- 本次先做“低风险一致性修复”，不做“大改 prompt 语义”。

## 本次改动的总体预期

1. 减少模型接收到的互相冲突指令。
2. 降低共享规则的维护成本。
3. 让 `prompts.py` 的运行时状态更清晰。
4. 在不改主流程接口的情况下，提高后续继续重构的可控性。

## 后续建议

如果继续优化，优先顺序建议是：

1. 把 4 个 `BASE_PROMPT_*` 的“工具协议”和“输出规范”继续抽成共享片段。
2. 把旧版拆分式 Prompt 挪到单独的 `legacy_prompts.py`。
3. 压缩 `UNIFIED_CLASSIFICATION_PROMPT`，减少首轮分类 token 成本。
4. 将 `<思考>` 区块从长自由文本改成更紧凑的结构化 scratchpad。
