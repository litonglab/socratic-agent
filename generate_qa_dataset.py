import os
import json

data_dir = "/Users/baoliliu/Downloads/networking-agent/RAG-Agent/data"
output_file = "test-rag/evaluation/qa_dataset.json"

doc_files = [f for f in os.listdir(data_dir) if f.endswith((".docx", ".doc"))]

qa_dataset = []

for doc_file in doc_files:
    doc_id = doc_file
    base_name = os.path.splitext(doc_file)[0]
    
    # 移除文件名中的年份信息，例如 "(2025版)"
    if "（" in base_name and "版）" in base_name:
        base_name = base_name.split("（")[0].strip()
    elif "(" in base_name and ")" in base_name:
        base_name = base_name.split("(")[0].strip()

    questions = [
        f"什么是{base_name}？",
        f"请介绍一下{base_name}的主要内容。",
        f"在{base_name}中，有哪些关键概念？"
    ]

    for q in questions:
        qa_dataset.append({
            "question": q,
            "ground_truth_answer": "", # 占位符
            "ground_truth_citations": [
                {"doc_id": doc_id, "section_path": ["概述"], "content_text_summary": f"来自文档 {doc_id} 的概述"}
            ]
        })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

print(f"生成的 QA 数据集已保存到: {output_file}")



