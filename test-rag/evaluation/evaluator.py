# test-rag/evaluation/evaluator.py

import os
import json
from typing import List, Dict, Any

class RAGEvaluator:
    def __init__(self, qa_dataset_path: str, rag_system_instance):
        self.qa_dataset = self._load_qa_dataset(qa_dataset_path)
        self.rag_system = rag_system_instance

    def _load_qa_dataset(self, qa_dataset_path: str) -> List[Dict]:
        """加载 QA 数据集，格式为 [{"question": "...", "ground_truth_answer": "...", "ground_truth_citations": [{"doc_id": "...", "section_path": [...]}]}]"""
        with open(qa_dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate(self) -> Dict:
        results = []
        for i, qa_entry in enumerate(self.qa_dataset):
            question = qa_entry["question"]
            ground_truth_answer = qa_entry.get("ground_truth_answer", "")
            ground_truth_citations = qa_entry.get("ground_truth_citations", [])

            # 模拟用户查询 RAG 系统
            rag_response = self.rag_system.query(question)
            
            generated_answer = rag_response["answer"]
            generated_citations = rag_response["citations"] # 假设 RAG 系统返回的引用格式与 ground_truth_citations 类似

            # 计算各项指标
            retrieval_metrics = self._evaluate_retrieval(qa_entry, rag_response)
            generation_metrics = self._evaluate_generation(qa_entry, rag_response)
            
            results.append({
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "generated_answer": generated_answer,
                "ground_truth_citations": ground_truth_citations,
                "generated_citations": generated_citations,
                **retrieval_metrics,
                **generation_metrics
            })
        
        # 聚合所有结果并计算平均值等
        aggregated_metrics = self._aggregate_results(results)
        return {"average_recall_at_k": aggregated_metrics["average_recall_at_k"], "average_faithfulness": aggregated_metrics["average_faithfulness"], "details": results}

    def _evaluate_retrieval(self, qa_entry: Dict, rag_response: Dict) -> Dict:
        """评估检索相关指标，如 Recall@K, MRR"""
        retrieved_chunks = rag_response.get("retrieved_chunks", [])
        ground_truth_citations = qa_entry.get("ground_truth_citations", [])

        recall_at_k = 0
        if ground_truth_citations and retrieved_chunks:
            for gt_citation in ground_truth_citations:
                for chunk in retrieved_chunks:
                    # 简化处理：检查 doc_id 是否匹配
                    if chunk["doc_id"] == gt_citation["doc_id"]:
                        recall_at_k = 1
                        break
                if recall_at_k == 1:
                    break
        
        return {"recall_at_k": recall_at_k}

    def _evaluate_generation(self, qa_entry: Dict, rag_response: Dict) -> Dict:
        """评估生成相关指标，如 Faithfulness, Answer Relevance"""
        generated_answer = rag_response["answer"]
        generated_citations = rag_response["citations"]
        ground_truth_answer = qa_entry.get("ground_truth_answer", "")

        faithfulness = 1 # 默认忠实
        if "我无法从提供的文档中找到相关信息" in generated_answer:
            if not ground_truth_answer and not generated_citations: # 期望拒答且正确拒答
                faithfulness = 1
            else: # 拒答但不应该拒答
                faithfulness = 0
        elif not generated_citations and ground_truth_answer: # 生成了答案，但没有引用，且有真实答案
            faithfulness = 0 # 假设为不忠实
        elif generated_citations and not ground_truth_answer: # 有引用但没有真实答案，并且也生成了答案
             faithfulness = 0 # 瞎编答案且有引用
        # 如果有引用且有真实答案，我们假设是忠实的，直到有更复杂的判断逻辑
        
        return {"faithfulness": faithfulness}

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """聚合所有结果并计算平均值等"""
        total_recall_at_k = sum([r["recall_at_k"] for r in results])
        avg_recall_at_k = total_recall_at_k / len(results) if results else 0

        total_faithfulness = sum([r["faithfulness"] for r in results])
        avg_faithfulness = total_faithfulness / len(results) if results else 0

        return {
            "average_recall_at_k": avg_recall_at_k,
            "average_faithfulness": avg_faithfulness,
        }




