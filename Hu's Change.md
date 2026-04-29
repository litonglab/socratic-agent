# Tool / Observation Message Refactor

## 背景

项目原先的 agent loop 在处理工具调用时，存在一个结构性问题：

1. 用户问题作为 `HumanMessage` 输入模型。
2. 模型输出 `<tool_calls>...</tool_calls>` 文本协议。
3. 系统执行工具后，将工具结果拼成一段文本。
4. 这段文本再次通过 `HumanMessage` 回灌给模型。

这会把“工具观察结果”伪装成“用户新增发言”。

原实现的关键路径在 [agentic_rag/agent.py](/home/octopus/socratic-agent/agentic_rag/agent.py)：

- `Agent.__call__()` 无条件将输入包装为 `HumanMessage`
- `query()` / `query_stream()` 在工具调用后把 `_execute_tool_actions()` 的返回值当成下一轮 `next_prompt`

## 为什么这是问题

虽然对模型来说最终都会被编码成 token，但消息 `role` 本身就是输入语义的一部分，不只是表现形式。

- `user` 表示用户在提问、补充需求或给出新信息
- `assistant` 表示模型之前自己的输出
- `tool` 表示外部工具返回的观察结果
- `system` 表示高优先级约束

因此，“用户说了什么”和“系统查到了什么”混在同一个 `user` 通道里，不是无害的实现细节，而是会削弱多轮工具推理的边界。

具体风险：

- 模型可能把工具结果误解为用户主张，而不是外部证据。
- 多轮工具链会丢失“assistant 发起调用 -> tool 返回观察 -> assistant 继续推理”的标准形态。
- 随着历史轮次增加，对话角色边界会越来越模糊。

## 本次改动

### 1. Agent 内部消息流改为标准三段式

修改文件：

- [agentic_rag/agent.py](/home/octopus/socratic-agent/agentic_rag/agent.py)

改动后的工具轮消息流如下：

1. `HumanMessage(question)`
2. `AIMessage(content=<tool_calls>..., tool_calls=[...])`
3. `ToolMessage(content=observation, tool_call_id=..., name=...)`
4. `AIMessage(final_answer)`

这意味着工具结果不再被重新包装成 `HumanMessage`。

### 2. 工具结果不再走 `next_prompt` 文本回灌

原先 `_execute_tool_actions()` 返回一整段字符串，随后被当成下一轮 user turn。

现在 `_execute_tool_actions()` 改为返回按顺序排列的 observation 文本列表，每条 observation 会被 append 成对应的 `ToolMessage`。

### 3. 为 assistant 补齐 `tool_calls` 元信息

当模型输出 `<tool_calls>` 后，系统会解析出每个工具动作，并为其构建稳定的 `tool_call_id`。

这样后续 `ToolMessage` 能与对应的 assistant tool call 对齐，而不是只有一段孤立文本。

### 4. DeepSeek 序列化层显式支持 `role="tool"`

修改文件：

- [agentic_rag/llm_config.py](/home/octopus/socratic-agent/agentic_rag/llm_config.py)

新增序列化逻辑：

- `ToolMessage` -> `{"role": "tool", "content": ..., "tool_call_id": ...}`
- `AIMessage(tool_calls=...)` -> 输出符合 DeepSeek chat completions 结构的 `tool_calls`

这一步是必要的。否则即便内部代码使用了 `ToolMessage`，最终发给模型时仍可能退化成 `role="user"`。

## 为什么没有直接切到 DeepSeek 原生 tools 参数

当前项目已经建立了一套文本协议：

- prompt 明确要求模型输出 `<tool_calls>...</tool_calls>`
- 普通与流式路径都围绕这个文本协议做了解析与截流

如果直接切到原生 `tools` / function calling：

- prompt 需要整体重写
- 流式解析逻辑要重做
- 兼容测试面显著扩大

这不是“局部修正消息边界”，而是“工具协议整体迁移”。

因此本次方案刻意采用折中但稳妥的路线：

- 保留现有 `<tool_calls>` 文本协议，避免大规模回归
- 在运行时消息图上恢复标准 `assistant -> tool -> assistant` 结构

这样既修复了角色语义错误，也控制了改动风险。

## 兼容性边界

### 会保留的行为

- prompt 仍要求模型输出 `<tool_calls>...</tool_calls>`
- 工具解析逻辑仍兼容现有 structured / legacy 两种解析方式
- `tool_traces` 仍保留，用于日志与前端展示

### 新增的行为

- DeepSeek 请求体里会出现 `role="tool"` 消息
- assistant 消息在工具轮会带 `tool_calls`

### 暂未扩展的部分

- 持久化历史和前端响应仍只保留 `user/assistant/system` 对话主链
- tool 消息目前用于运行时推理，不作为最终对话历史对外暴露

这是有意控制的范围。因为用户界面并不需要直接展示 tool 消息原文。

## 验证

本次改动完成后做了以下验证：

1. `python -m py_compile agentic_rag/agent.py agentic_rag/llm_config.py`
2. 本地构造 `AIMessage(tool_calls=...)` 与 `ToolMessage(...)`，确认序列化结果包含：
   - assistant `tool_calls`
   - tool `role`
   - `tool_call_id`

## 后续建议

如果后续希望进一步标准化工具调用，可以考虑第二阶段改造：

1. 将 prompts 从 `<tool_calls>` 文本协议迁移到原生 tool calling
2. 让流式路径直接处理原生增量 tool call chunk
3. 为 `messages_to_dicts()` / `dicts_to_messages()` 增加更完整的 tool 元数据持久化策略

这属于独立课题，不建议和本次“修正角色语义边界”的改动混做。

# Hint Level 状态机重构

## 背景

原先 `hint_level` 的升级逻辑主要依赖两条规则：

1. 让 LLM 输出 `MAINTAIN / INCREASE / JUMP_TO_MAX`
2. 若连续 3 轮没有触发 `INCREASE`，则用 failsafe 强制升级

对于 `LAB_TROUBLESHOOTING`，还额外叠加了一条硬规则：

1. 只要连续 3 轮都被分到 `LAB_TROUBLESHOOTING`
2. 就直接把 `hint_level` 拉到该类别最大值
3. 同时 prompt 注入“本轮必须收敛，不依赖学生继续回答”

这套逻辑实现简单，但判断依据并不严谨。

## 为什么这是问题

问题不在于“用了 LLM”，而在于运行时没有把教学状态显式建模出来。

原实现虽然让 LLM 在 prompt 里考虑“是否有进展”，但代码层只真正持久化了：

- 当前 `hint_level`
- 当前等级已持续多少轮
- `LAB_TROUBLESHOOTING` 连续出现多少轮

没有显式记录这些更可靠的运行时信号：

- 学生是否明确表达“不会 / 看不懂 / 直接给答案”
- 学生是否已经贴出关键命令输出
- 学生是否补充了新的故障现象
- 学生是否给出了拓扑 / 设备 / 接口上下文
- 问题是否已经切换到别的子话题
- 当前排障是否已经收敛或解决

结果就是：轮数本身在决策里占了过大的比重。

这会带来两个具体偏差：

1. “对话变长”容易被误判成“学生卡住了”
2. 实验排障会因为轮数达标而被过早推向“直接收敛”

对于复杂实验排障，这两个偏差都不理想。因为真实排障常常需要先收集现象、再补命令输出、再对照拓扑、最后才进入根因收敛。

## 本次改动

### 1. 从“单一计数器”改成“显式信号 + 小状态机”

修改文件：

- [agentic_rag/agent.py](/home/octopus/socratic-agent/agentic_rag/agent.py)

新增 `HintSignals` 和状态字段，用来显式区分：

- 卡住信号：`explicit_confusion`、`frustration`、`repeated_reply`
- 进展信号：`has_new_evidence`、`solved`
- 话题漂移信号：`topic_shift`
- 收敛条件：`evidence_complete`

并新增持久化在 `state` 中的运行时字段：

- `hint_state_phase`
- `hint_stagnation_turns`
- `lab_evidence_score`
- `lab_evidence_flags`

其中 `hint_state_phase` 是一个小阶段机，而不是单纯轮数：

- `probing`
- `gathering_evidence`
- `narrowing_root_cause`
- `proposing_fix`
- `resolved`

这意味着系统现在会先判断“排障走到哪一阶段了”，再决定是否升级，而不是简单看“已经聊了几轮”。

### 2. 为 LAB 场景引入“证据完备度”而不是“三轮硬收敛”

原先 `LAB_TROUBLESHOOTING` 只要累计到第 3 轮，就直接：

- `hint_level = max_level`
- prompt 注入“本轮必须收敛”

这个规则现在被删除。

替代方案是对实验排障做一个轻量证据评分，当前主要看四类证据是否出现：

- 故障现象
- 命令输出 / 观察结果
- 拓扑或设备上下文
- 已执行的排查动作

当这些信号基本齐全时，状态机会把阶段推进到 `proposing_fix`，只有这时 prompt 才会被追加“证据驱动收敛”约束。

也就是说，系统从“轮数驱动收敛”改成了“证据驱动收敛”。

### 3. 把 failsafe 从“无条件 3 轮升级”改成“分阶段卡住才升级”

原实现中，只要连续 3 轮没有 `INCREASE`，就会强制 `hint_level + 1`。

这条规则的问题是：它并不区分学生是在推进、补证据、换子问题，还是确实在原地打转。

新实现里，failsafe 只会在显式“卡住”时触发，而且按阶段区分阈值：

- `probing` 阶段卡住 3 轮，才允许升级
- `gathering_evidence` 阶段卡住 4 轮，才允许升级
- 已进入 `narrowing_root_cause` 且学生明确困惑，才更积极升级

这样做的意图是：

- 在证据还不够时，允许系统继续追证而不是急着放弃引导
- 在已经具备定位基础时，再更果断地提升帮助强度

### 4. 保留 LLM 作为软信号，但不再让它单独支配升级

LLM 的 `hint_decision` 没有被删除，但其角色被收敛成“软信号”。

现在升级必须和运行时信号互相印证。例如：

- `INCREASE` + 学生明确表示不会 / 挫败
- `INCREASE` + 当前 LAB 阶段没有新证据且停滞
- `JUMP_TO_MAX` + 用户明确要求直接给答案

这比单纯信任一轮 LLM 分类更稳，也更可解释。

### 5. 同步收紧 unified classification prompt

修改文件：

- [agentic_rag/prompts.py](/home/octopus/socratic-agent/agentic_rag/prompts.py)

更新后的 prompt 明确要求分类器：

- 不要因为轮数增加就自动升级
- 对 `LAB_TROUBLESHOOTING` 优先看证据完备度
- “没有实质进展”的定义应绑定到“没有新证据 / 新信息”
- 即使轮数增加，只要学生提供了新输出、新现象或新拓扑信息，也应倾向 `MAINTAIN`

这一步是必要的。否则运行时代码虽然变严谨了，LLM 仍可能持续给出偏激进的 `INCREASE` 倾向。

## 为什么这次改动更严谨，而不只是更稳妥

这次方案不是简单把阈值从 3 改到 4，也不是只把 prompt 写得更保守。

真正的变化在于：系统终于区分了“对话轮数”和“教学状态”。

换句话说，升级逻辑现在回答的是这些问题：

1. 学生是否真的卡住了？
2. 学生是否拿来了新的证据？
3. 是不是已经切换到别的问题？
4. 当前 LAB 排障是否已经具备收敛条件？

而不是只回答：

1. 这一等级聊了几轮？
2. 这是不是 LAB 的第 3 轮？

前者是状态建模，后者只是计数。

## 兼容性与影响

### 保留的行为

- 仍然保留 `hint_level` 作为外部教学强度控制变量
- 仍然保留 `hint_decision` 的埋点，兼容现有用户水平统计逻辑
- 仍然保留 `turns_at_current_level` 和 `lab_turn_count`，避免其他代码路径断裂

### 新增的运行时信息

- `_hint_transition_reason`
- `_hint_phase`
- `_hint_evidence_score`
- `_hint_stagnation_turns`

这些字段目前主要用于日志和后续观测，没有强耦合到存储 schema，因此改动范围受控。

## 验证

本次改动完成后做了以下验证：

1. `python -m py_compile agentic_rag/agent.py agentic_rag/prompts.py`

## 后续建议

当前的证据评分还是基于规则的轻量启发式，不是最终形态。

如果后续继续深化，可以考虑：

1. 将“要求执行的关键命令”显式写入状态，并检测学生是否真的返回了对应输出
2. 将 `ToolMessage` 中的工具观察结果也纳入证据完备度，而不只看用户文本
3. 为 `LAB_TROUBLESHOOTING` 单独记录“已排除的假设集合”，让状态机从“阶段机”进一步演进为“诊断图”

这会让 hint 升级从“教学强度控制”进一步走向“可观测诊断流程控制”。

## 对上一版状态机的修正

在第一版状态机落地后，又补做了一轮实现审查，发现其中有三处会削弱“严谨且不失稳妥”的目标，因此本轮继续修正。

### 1. 将 LAB 证据从布尔命中改为累积槽位

上一版只做了 4 个布尔 flag：

- symptom
- output
- topology
- action

并通过最近几条用户消息上的正则命中来计算 `evidence_score`。

这个做法的问题是：

1. 学生贴出了新的命令输出，但如果仍然落在同一类 flag，系统看不出“这是新证据”
2. 证据判断依赖短窗口文本，旧证据可能因为不再重复出现而“消失”

本轮改成了结构化槽位累积：

- `lab_evidence_slots["symptom"]`
- `lab_evidence_slots["output"]`
- `lab_evidence_slots["topology"]`
- `lab_evidence_slots["action"]`

每轮只提取本轮新增的片段，再 merge 到状态里。这样后续判断的是：

- 某类证据是否已经收集过
- 这一轮是否真正新增了证据

而不是只看“最近两条消息有没有命中关键词”。

### 2. 将“证据完备”从关键词收敛改为真实输出约束

上一版存在一个偏差：只要文本里出现了 `show` / `display` / `ospf` / `vlan` 之类词，就可能让 `output` 或 `topology` 命中，从而过早满足 `evidence_complete`。

这会把“证据驱动收敛”重新退化成“关键词驱动收敛”。

本轮修正后：

- `output` 槽位不再因为“提到过命令名”就成立
- 只有满足“命令/输出标记 + 结果内容”或直接贴出代码块时，才认为拿到了真实输出
- `evidence_complete` 现在要求至少具备：
  - 故障现象
  - 真实输出
  - 拓扑/设备上下文

这样进入 `proposing_fix` 的门槛更接近真实排障所需的信息完整度。

### 3. 将 proficiency 埋点从旧 LLM 语义补齐到新状态机语义

上一版虽然已经把真实升级决策交给状态机，但 `storage/proficiency.py` 仍主要依赖：

- `_hint_decision`
- `_was_failsafe`

这会导致画像层误把“规则收敛”“用户要求直接给答案”等情形当成单纯的能力下降。

本轮把这些新状态机信息纳入评分：

- `_hint_transition_reason`
- `_hint_phase`
- `_hint_evidence_score`
- `_hint_stagnation_turns`

并做了更细的区分：

- `direct_answer_request` 不应直接视作能力差
- `evidence_complete_ready_to_converge` 与 `resolved` 应有正向信号
- `stalled_without_evidence` / `evidence_collection_stalled` 才是更明确的负向信号

这让学生画像开始和新的 hint 状态机对齐，而不是继续解释旧世界里的 `MAINTAIN / INCREASE`。

## 本轮验证

本轮修正完成后做了以下验证：

1. `python -m py_compile agentic_rag/agent.py storage/proficiency.py`
