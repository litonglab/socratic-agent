<mxfile host="app.diagrams.net" modified="2026-03-11T11:20:00.000Z" agent="GPT-5.4" version="24.7.17">
  <diagram id="fig5-1-system-runtime-clean" name="图5-1 系统实现组成与运行关系图（精简版）">
    <mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1800" pageHeight="920" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>

        <mxCell id="2" value="图5-1 系统实现组成与运行关系图" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#0F172A;" vertex="1" parent="1">
          <mxGeometry x="525" y="18" width="750" height="36" as="geometry"/>
        </mxCell>
        <mxCell id="3" value="以运行主链为中心，展示系统的核心实现模块及知识、状态与外部依赖之间的支撑关系" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=13;fontColor=#475569;" vertex="1" parent="1">
          <mxGeometry x="320" y="56" width="1160" height="24" as="geometry"/>
        </mxCell>

        <mxCell id="10" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=14;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="60" y="110" width="1680" height="250" as="geometry"/>
        </mxCell>
        <mxCell id="11" value="运行主链" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#334155;" vertex="1" parent="1">
          <mxGeometry x="85" y="126" width="140" height="24" as="geometry"/>
        </mxCell>

        <mxCell id="20" value="学生 / 教师" style="ellipse;whiteSpace=wrap;html=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=2;fontSize=16;fontStyle=1;fontColor=#1E3A8A;" vertex="1" parent="1">
          <mxGeometry x="90" y="205" width="130" height="60" as="geometry"/>
        </mxCell>

        <mxCell id="21" value="&lt;b&gt;前端交互&lt;/b&gt;&lt;br&gt;登录、会话、图片输入、反馈" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=2;fontSize=14;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="290" y="185" width="220" height="100" as="geometry"/>
        </mxCell>

        <mxCell id="22" value="&lt;b&gt;服务接口&lt;/b&gt;&lt;br&gt;认证、聊天、流式返回、会话读写" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#ECFDF5;strokeColor=#10B981;strokeWidth=2;fontSize=14;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="585" y="185" width="230" height="100" as="geometry"/>
        </mxCell>

        <mxCell id="23" value="&lt;b&gt;Agent 调度&lt;/b&gt;&lt;br&gt;问题理解、分类、Hint 调节、工具路由" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#FFF7ED;strokeColor=#F97316;strokeWidth=2;fontSize=14;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="890" y="175" width="260" height="120" as="geometry"/>
        </mxCell>

        <mxCell id="24" value="&lt;b&gt;知识与工具调用&lt;/b&gt;&lt;br&gt;文本 RAG、拓扑检索、OCR、联网搜索" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#F5F3FF;strokeColor=#8B5CF6;strokeWidth=2;fontSize=14;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1225" y="175" width="260" height="120" as="geometry"/>
        </mxCell>

        <mxCell id="25" value="&lt;b&gt;回答生成与回传&lt;/b&gt;&lt;br&gt;证据约束生成、流式输出" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#DCFCE7;strokeColor=#16A34A;strokeWidth=2;fontSize=14;fontColor=#14532D;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1560" y="185" width="150" height="100" as="geometry"/>
        </mxCell>

        <mxCell id="30" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=14;fillColor=#FFFBEB;strokeColor=#F59E0B;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="60" y="410" width="980" height="220" as="geometry"/>
        </mxCell>
        <mxCell id="31" value="知识与状态支撑" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#92400E;" vertex="1" parent="1">
          <mxGeometry x="85" y="426" width="180" height="24" as="geometry"/>
        </mxCell>

        <mxCell id="32" value="&lt;b&gt;课程文档与文本索引&lt;/b&gt;&lt;br&gt;课程资料、分块文本、FAISS" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="100" y="485" width="230" height="78" as="geometry"/>
        </mxCell>
        <mxCell id="33" value="&lt;b&gt;拓扑结构化结果&lt;/b&gt;&lt;br&gt;approved JSON、实验映射信息" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="365" y="485" width="230" height="78" as="geometry"/>
        </mxCell>
        <mxCell id="34" value="&lt;b&gt;用户与会话数据库&lt;/b&gt;&lt;br&gt;用户、session、feedback、日志" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="630" y="485" width="230" height="78" as="geometry"/>
        </mxCell>
        <mxCell id="35" value="&lt;b&gt;学习者状态&lt;/b&gt;&lt;br&gt;摘要、指标、能力分数与个性化调节" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#F59E0B;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="365" y="575" width="330" height="40" as="geometry"/>
        </mxCell>

        <mxCell id="40" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=14;fillColor=#F8FAFC;strokeColor=#64748B;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="1095" y="410" width="645" height="220" as="geometry"/>
        </mxCell>
        <mxCell id="41" value="外部依赖与离线构建" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=18;fontStyle=1;fontColor=#334155;" vertex="1" parent="1">
          <mxGeometry x="1120" y="426" width="220" height="24" as="geometry"/>
        </mxCell>

        <mxCell id="42" value="&lt;b&gt;大语言模型服务&lt;/b&gt;&lt;br&gt;问题理解与回答生成" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1135" y="490" width="180" height="78" as="geometry"/>
        </mxCell>
        <mxCell id="43" value="&lt;b&gt;嵌入与重排模型&lt;/b&gt;&lt;br&gt;支持向量召回与相关性优化" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1340" y="490" width="180" height="78" as="geometry"/>
        </mxCell>
        <mxCell id="44" value="&lt;b&gt;离线知识构建脚本&lt;/b&gt;&lt;br&gt;文档分块、拓扑抽取、审核发布" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.6;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1545" y="490" width="180" height="78" as="geometry"/>
        </mxCell>

        <mxCell id="50" value="输入" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="20" target="21">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="51" value="请求" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="21" target="22">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="52" value="上下文" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="22" target="23">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="53" value="调用" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="23" target="24">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="54" value="证据" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;fontSize=12;" edge="1" parent="1" source="24" target="25">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="55" value="流式回传" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#16A34A;fontSize=12;" edge="1" parent="1" source="25" target="21">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <mxCell id="56" value="文本知识支撑" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;fontSize=12;" edge="1" parent="1" source="32" target="24">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="57" value="拓扑支撑" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;fontSize=12;" edge="1" parent="1" source="33" target="24">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="58" value="会话读写" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;fontSize=12;" edge="1" parent="1" source="34" target="22">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="59" value="状态反馈" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;dashed=1;dashPattern=7 5;fontSize=12;" edge="1" parent="1" source="35" target="23">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <mxCell id="60" value="模型调用" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;fontSize=12;" edge="1" parent="1" source="42" target="23">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="61" value="检索优化" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;fontSize=12;" edge="1" parent="1" source="43" target="32">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="62" value="离线构建" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;fontSize=12;" edge="1" parent="1" source="44" target="32">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="63" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;" edge="1" parent="1" source="44" target="33">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <mxCell id="64" value="注：该图强调系统在运行过程中的实现组成与关系，其中上方为在线处理主链，下方为知识、状态和外部依赖支撑。" style="shape=note;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;fontSize=12;fontColor=#475569;spacing=8;" vertex="1" parent="1">
          <mxGeometry x="60" y="675" width="1680" height="48" as="geometry"/>
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
