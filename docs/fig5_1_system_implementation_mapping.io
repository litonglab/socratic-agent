<mxfile host="app.diagrams.net" modified="2026-03-11T10:30:00.000Z" agent="GPT-5.4" version="24.7.17">
  <diagram id="fig5-1-system-implementation-mapping" name="图5-1 系统总体架构与实现模块对应图">
    <mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1800" pageHeight="1000" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>

        <mxCell id="2" value="图5-1 系统总体架构与实现模块对应图" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=22;fontStyle=1;fontColor=#0F172A;" vertex="1" parent="1">
          <mxGeometry x="520" y="20" width="760" height="36" as="geometry"/>
        </mxCell>
        <mxCell id="3" value="在第四章总体架构基础上，从工程实现角度说明各功能层的具体落地模块及其协同关系" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=13;fontColor=#475569;" vertex="1" parent="1">
          <mxGeometry x="340" y="58" width="1120" height="24" as="geometry"/>
        </mxCell>

        <mxCell id="4" value="学生 / 教师" style="ellipse;whiteSpace=wrap;html=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=2;fontSize=16;fontStyle=1;fontColor=#1E3A8A;" vertex="1" parent="1">
          <mxGeometry x="70" y="365" width="140" height="60" as="geometry"/>
        </mxCell>

        <mxCell id="10" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="250" y="250" width="300" height="240" as="geometry"/>
        </mxCell>
        <mxCell id="11" value="交互层" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=17;fontStyle=1;fontColor=#1D4ED8;" vertex="1" parent="1">
          <mxGeometry x="275" y="265" width="120" height="24" as="geometry"/>
        </mxCell>
        <mxCell id="12" value="&lt;b&gt;Streamlit 会话界面&lt;/b&gt;&lt;br&gt;登录注册 / 会话切换 / 问答展示" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#60A5FA;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="285" y="305" width="230" height="56" as="geometry"/>
        </mxCell>
        <mxCell id="13" value="&lt;b&gt;图片输入与反馈交互&lt;/b&gt;&lt;br&gt;图片上传 / 粘贴输入 / 点赞点踩" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#60A5FA;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="285" y="378" width="230" height="56" as="geometry"/>
        </mxCell>
        <mxCell id="14" value="&lt;b&gt;OCR 文本融合&lt;/b&gt;&lt;br&gt;将图片文字提取后并入用户问题" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#60A5FA;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="285" y="451" width="230" height="56" as="geometry"/>
        </mxCell>

        <mxCell id="20" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#ECFDF5;strokeColor=#10B981;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="590" y="250" width="300" height="240" as="geometry"/>
        </mxCell>
        <mxCell id="21" value="服务层" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=17;fontStyle=1;fontColor=#047857;" vertex="1" parent="1">
          <mxGeometry x="615" y="265" width="120" height="24" as="geometry"/>
        </mxCell>
        <mxCell id="22" value="&lt;b&gt;FastAPI 服务入口&lt;/b&gt;&lt;br&gt;统一接收聊天、认证与会话请求" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#34D399;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="625" y="305" width="230" height="56" as="geometry"/>
        </mxCell>
        <mxCell id="23" value="&lt;b&gt;认证与会话管理&lt;/b&gt;&lt;br&gt;用户身份、历史记录与上下文组织" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#34D399;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="625" y="378" width="230" height="56" as="geometry"/>
        </mxCell>
        <mxCell id="24" value="&lt;b&gt;流式响应与持久化&lt;/b&gt;&lt;br&gt;答案回传、摘要更新与日志记录" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#34D399;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="625" y="451" width="230" height="56" as="geometry"/>
        </mxCell>

        <mxCell id="30" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#FFF7ED;strokeColor=#F97316;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="930" y="250" width="360" height="240" as="geometry"/>
        </mxCell>
        <mxCell id="31" value="Agent 编排层" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=17;fontStyle=1;fontColor=#C2410C;" vertex="1" parent="1">
          <mxGeometry x="955" y="265" width="160" height="24" as="geometry"/>
        </mxCell>
        <mxCell id="32" value="&lt;b&gt;问题理解与分类&lt;/b&gt;&lt;br&gt;相关性判断 / 任务类型识别" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#FB923C;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="955" y="305" width="145" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="33" value="&lt;b&gt;Hint Level 控制&lt;/b&gt;&lt;br&gt;分层引导 / 支持强度调节" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#FB923C;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1110" y="305" width="145" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="34" value="&lt;b&gt;Prompt 构建与工具路由&lt;/b&gt;&lt;br&gt;整合问题、上下文与教学约束" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#FB923C;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="955" y="395" width="145" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="35" value="&lt;b&gt;Agent 执行循环&lt;/b&gt;&lt;br&gt;调用工具、整理证据并生成回答" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#FB923C;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1110" y="395" width="145" height="70" as="geometry"/>
        </mxCell>

        <mxCell id="40" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#F5F3FF;strokeColor=#8B5CF6;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="1330" y="230" width="410" height="280" as="geometry"/>
        </mxCell>
        <mxCell id="41" value="检索、教学与评审层" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=17;fontStyle=1;fontColor=#6D28D9;" vertex="1" parent="1">
          <mxGeometry x="1355" y="245" width="220" height="24" as="geometry"/>
        </mxCell>
        <mxCell id="42" value="&lt;b&gt;文本 RAG 检索&lt;/b&gt;&lt;br&gt;课程文档召回与引用生成" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#A78BFA;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1355" y="285" width="160" height="64" as="geometry"/>
        </mxCell>
        <mxCell id="43" value="&lt;b&gt;拓扑知识检索&lt;/b&gt;&lt;br&gt;读取审核通过的拓扑 JSON" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#A78BFA;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1535" y="285" width="160" height="64" as="geometry"/>
        </mxCell>
        <mxCell id="44" value="&lt;b&gt;联网搜索补充&lt;/b&gt;&lt;br&gt;课程外知识扩展与兜底" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#A78BFA;strokeWidth=1.5;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1355" y="370" width="160" height="64" as="geometry"/>
        </mxCell>
        <mxCell id="45" value="&lt;b&gt;教学回答生成&lt;/b&gt;&lt;br&gt;结合证据与教学策略输出结果" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#DCFCE7;strokeColor=#16A34A;strokeWidth=1.5;fontSize=13;fontColor=#14532D;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1535" y="370" width="160" height="64" as="geometry"/>
        </mxCell>
        <mxCell id="46" value="&lt;b&gt;评审能力预留&lt;/b&gt;&lt;br&gt;候选答案比较与结果筛选" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#7C3AED;strokeWidth=1.5;dashed=1;dashPattern=6 4;fontSize=13;fontColor=#1F2937;spacingTop=6;" vertex="1" parent="1">
          <mxGeometry x="1445" y="455" width="160" height="44" as="geometry"/>
        </mxCell>

        <mxCell id="50" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#F8FAFC;strokeColor=#64748B;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="930" y="100" width="810" height="90" as="geometry"/>
        </mxCell>
        <mxCell id="51" value="外部模型与服务" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#334155;" vertex="1" parent="1">
          <mxGeometry x="955" y="115" width="180" height="22" as="geometry"/>
        </mxCell>
        <mxCell id="52" value="大语言模型服务" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.4;fontSize=13;fontColor=#1F2937;" vertex="1" parent="1">
          <mxGeometry x="980" y="145" width="170" height="34" as="geometry"/>
        </mxCell>
        <mxCell id="53" value="嵌入与重排模型" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.4;fontSize=13;fontColor=#1F2937;" vertex="1" parent="1">
          <mxGeometry x="1248" y="145" width="170" height="34" as="geometry"/>
        </mxCell>
        <mxCell id="54" value="外部搜索服务" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.4;fontSize=13;fontColor=#1F2937;" vertex="1" parent="1">
          <mxGeometry x="1515" y="145" width="170" height="34" as="geometry"/>
        </mxCell>

        <mxCell id="60" value="" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#FEFCE8;strokeColor=#EAB308;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="1330" y="560" width="410" height="150" as="geometry"/>
        </mxCell>
        <mxCell id="61" value="离线知识构建通道" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#A16207;" vertex="1" parent="1">
          <mxGeometry x="1355" y="575" width="190" height="22" as="geometry"/>
        </mxCell>
        <mxCell id="62" value="原始课程资源" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#EAB308;strokeWidth=1.4;fontSize=13;fontColor=#1F2937;" vertex="1" parent="1">
          <mxGeometry x="1355" y="615" width="110" height="42" as="geometry"/>
        </mxCell>
        <mxCell id="63" value="文本分块与向量索引构建" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#EAB308;strokeWidth=1.4;fontSize=13;fontColor=#1F2937;" vertex="1" parent="1">
          <mxGeometry x="1480" y="615" width="120" height="42" as="geometry"/>
        </mxCell>
        <mxCell id="64" value="拓扑抽取、分类与审核" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#FFFFFF;strokeColor=#EAB308;strokeWidth=1.4;fontSize=13;fontColor=#1F2937;" vertex="1" parent="1">
          <mxGeometry x="1615" y="615" width="110" height="42" as="geometry"/>
        </mxCell>

        <mxCell id="70" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;" edge="1" parent="1" source="4" target="10">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="71" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;" edge="1" parent="1" source="10" target="20">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="72" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;" edge="1" parent="1" source="20" target="30">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="73" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor=#475569;" edge="1" parent="1" source="30" target="40">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="74" value="模型与搜索支撑" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;fontSize=12;" edge="1" parent="1" source="50" target="30">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="75" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#64748B;" edge="1" parent="1" source="50" target="40">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="76" value="知识底座支撑" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#A16207;fontSize=12;" edge="1" parent="1" source="60" target="40">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
