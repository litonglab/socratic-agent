<mxfile host="app.diagrams.net" modified="2026-03-11T11:05:00.000Z" agent="GPT-5.4" version="24.7.17">
  <diagram id="fig5-1-system-runtime" name="图5-1 系统实现组成与运行关系图">
    <mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1800" pageHeight="980" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>

        <mxCell id="2" value="图5-1 系统实现组成与运行关系图" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#0F172A;" vertex="1" parent="1">
          <mxGeometry x="530" y="18" width="740" height="36" as="geometry"/>
        </mxCell>
        <mxCell id="3" value="从运行过程角度展示系统的核心组成、调用关系以及知识与状态支撑方式" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=13;fontColor=#475569;" vertex="1" parent="1">
          <mxGeometry x="350" y="56" width="1100" height="24" as="geometry"/>
        </mxCell>

        <mxCell id="10" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=14;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="55" y="110" width="1420" height="310" as="geometry"/>
        </mxCell>
        <mxCell id="11" value="运行主链" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#334155;" vertex="1" parent="1">
          <mxGeometry x="80" y="126" width="140" height="24" as="geometry"/>
        </mxCell>

        <mxCell id="20" value="学生 / 教师" style="ellipse;whiteSpace=wrap;html=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=2;fontSize=16;fontStyle=1;fontColor=#1E3A8A;" vertex="1" parent="1">
          <mxGeometry x="90" y="228" width="130" height="60" as="geometry"/>
        </mxCell>

        <mxCell id="21" value="&lt;b&gt;前端交互界面&lt;/b&gt;&lt;br&gt;Streamlit&lt;br&gt;登录、会话切换、图片输入、反馈操作" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=2;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="260" y="198" width="200" height="120" as="geometry"/>
        </mxCell>

        <mxCell id="22" value="&lt;b&gt;服务接口层&lt;/b&gt;&lt;br&gt;FastAPI&lt;br&gt;聊天、认证、流式返回、会话读写" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#ECFDF5;strokeColor=#10B981;strokeWidth=2;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="515" y="198" width="200" height="120" as="geometry"/>
        </mxCell>

        <mxCell id="23" value="&lt;b&gt;Agent 调度核心&lt;/b&gt;&lt;br&gt;问题理解、分类、Hint 调节&lt;br&gt;Prompt 组装与工具调度" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#FFF7ED;strokeColor=#F97316;strokeWidth=2;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="770" y="188" width="230" height="140" as="geometry"/>
        </mxCell>

        <mxCell id="24" value="&lt;b&gt;工具能力层&lt;/b&gt;&lt;br&gt;文本 RAG / 拓扑检索 / OCR / 联网搜索&lt;br&gt;按需获取证据与补充信息" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#F5F3FF;strokeColor=#8B5CF6;strokeWidth=2;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1055" y="188" width="240" height="140" as="geometry"/>
        </mxCell>

        <mxCell id="25" value="&lt;b&gt;回答生成与回传&lt;/b&gt;&lt;br&gt;整合证据、生成教学回答&lt;br&gt;流式输出给前端" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#DCFCE7;strokeColor=#16A34A;strokeWidth=2;fontSize=13;fontColor=#14532D;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1340" y="198" width="200" height="120" as="geometry"/>
        </mxCell>

        <mxCell id="30" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=14;fillColor=#FFFBEB;strokeColor=#F59E0B;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="55" y="465" width="980" height="250" as="geometry"/>
        </mxCell>
        <mxCell id="31" value="知识与状态支撑" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#92400E;" vertex="1" parent="1">
          <mxGeometry x="80" y="482" width="180" height="24" as="geometry"/>
        </mxCell>

        <mxCell id="32" value="&lt;b&gt;课程文档知识库&lt;/b&gt;&lt;br&gt;实验指导书、课程资料、文本块" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="90" y="540" width="180" height="72" as="geometry"/>
        </mxCell>
        <mxCell id="33" value="&lt;b&gt;FAISS 文本索引&lt;/b&gt;&lt;br&gt;向量召回、引用定位" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="300" y="540" width="180" height="72" as="geometry"/>
        </mxCell>
        <mxCell id="34" value="&lt;b&gt;拓扑结构化结果&lt;/b&gt;&lt;br&gt;approved JSON、实验编号映射" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="510" y="540" width="190" height="72" as="geometry"/>
        </mxCell>
        <mxCell id="35" value="&lt;b&gt;用户与会话数据库&lt;/b&gt;&lt;br&gt;用户、token、session、message feedback" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="730" y="540" width="220" height="72" as="geometry"/>
        </mxCell>
        <mxCell id="36" value="&lt;b&gt;交互指标与学习者状态&lt;/b&gt;&lt;br&gt;摘要、日志、能力分数、Hint 调节依据" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="300" y="630" width="400" height="72" as="geometry"/>
        </mxCell>

        <mxCell id="40" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=14;fillColor=#F8FAFC;strokeColor=#64748B;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="1085" y="465" width="390" height="250" as="geometry"/>
        </mxCell>
        <mxCell id="41" value="外部依赖与离线构建" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#334155;" vertex="1" parent="1">
          <mxGeometry x="1110" y="482" width="220" height="24" as="geometry"/>
        </mxCell>

        <mxCell id="42" value="&lt;b&gt;大语言模型服务&lt;/b&gt;&lt;br&gt;理解问题、生成回答" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1115" y="540" width="150" height="72" as="geometry"/>
        </mxCell>
        <mxCell id="43" value="&lt;b&gt;嵌入 / 重排模型&lt;/b&gt;&lt;br&gt;支持文本召回与精排" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1285" y="540" width="150" height="72" as="geometry"/>
        </mxCell>
        <mxCell id="44" value="&lt;b&gt;离线知识构建脚本&lt;/b&gt;&lt;br&gt;文档分块、拓扑抽取、审核发布" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1200" y="635" width="150" height="72" as="geometry"/>
        </mxCell>

        <mxCell id="50" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;" edge="1" parent="1" source="20" target="21">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="51" value="输入问题、图片与操作" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="20" target="21">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="52" value="请求" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="21" target="22">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="53" value="上下文与会话信息" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="22" target="23">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="54" value="工具调用" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="23" target="24">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="55" value="证据与中间结果" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="24" target="25">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="56" value="流式回答" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#16A34A;fontSize=12;" edge="1" parent="1" source="25" target="21">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <mxCell id="57" value="文本检索" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;fontSize=12;" edge="1" parent="1" source="33" target="24">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="58" value="拓扑检索" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;fontSize=12;" edge="1" parent="1" source="34" target="24">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="59" value="会话读写" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;fontSize=12;" edge="1" parent="1" source="35" target="22">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="60" value="状态更新" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;fontSize=12;" edge="1" parent="1" source="25" target="36">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="61" value="个性化调节" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;dashed=1;dashPattern=7 5;fontSize=12;" edge="1" parent="1" source="36" target="23">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <mxCell id="62" value="模型调用" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;fontSize=12;" edge="1" parent="1" source="42" target="23">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="63" value="向量化/精排" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;fontSize=12;" edge="1" parent="1" source="43" target="33">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="64" value="离线构建" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;fontSize=12;" edge="1" parent="1" source="44" target="32">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="65" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;" edge="1" parent="1" source="44" target="33">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="66" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;" edge="1" parent="1" source="44" target="34">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <mxCell id="67" value="注：上方展示系统运行主链，下方展示知识、状态和外部依赖等支撑关系；虚线表示状态反馈对后续交互的调节。" style="shape=note;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;fontSize=12;fontColor=#475569;spacing=8;" vertex="1" parent="1">
          <mxGeometry x="55" y="760" width="1420" height="52" as="geometry"/>
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
