
from sentence_transformers import CrossEncoder
from typing import List, Dict
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# class SpladeRetriever:
#     def __init__(self):
#         self.model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
#         self.tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
#
#     def encode(self, text: str) -> dict:
#         """生成Term权重字典"""
#         inputs = self.tokenizer(text, return_tensors="pt")
#         with torch.no_grad():
#             logits = self.model(**inputs).logits
#         # 提取重要Term及其权重（稀疏向量）
#         return {self.tokenizer.decode(idx): float(val)
#                 for idx, val in zip(logits.nonzero(), logits[logits > 0])}

# still need investation

class Reranker:
    """两阶段重排序器"""

    def __init__(self):
        # 第一阶段：轻量级粗排模型
        self.coarse_ranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

        # 第二阶段：精细排序模型
        self.fine_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # 业务规则过滤器
        self.business_rules = [
            lambda doc: "过期" not in doc["text"],  # 示例规则
            lambda doc: len(doc["text"]) > 20
        ]

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """执行两阶段重排序"""
        if not candidates:
            return []

        # 第一阶段：规则过滤 + 粗排
        filtered = self._apply_business_rules(candidates)
        coarse_ranked = self._coarse_ranking(query, filtered[:500])  # 限制数量

        # 第二阶段：精细排序
        fine_ranked = self._fine_ranking(query, coarse_ranked[:100])

        return fine_ranked

    def _apply_business_rules(self, docs: List[Dict]) -> List[Dict]:
        """应用业务规则过滤"""
        return [doc for doc in docs if all(rule(doc) for rule in self.business_rules)]

    def _coarse_ranking(self, query: str, docs: List[Dict]) -> List[Dict]:
        """轻量级粗排"""
        if not docs:
            return []

        model_inputs = [(query, doc["text"]) for doc in docs]
        scores = self.coarse_ranker.predict(model_inputs)

        for doc, score in zip(docs, scores):
            doc["coarse_score"] = float(score)

        return sorted(docs, key=lambda x: x["coarse_score"], reverse=True)

    def _fine_ranking(self, query: str, docs: List[Dict]) -> List[Dict]:
        """精细排序"""
        if len(docs) <= 1:
            return docs

        model_inputs = [(query, doc["text"]) for doc in docs]
        scores = self.fine_ranker.predict(model_inputs)

        for doc, score in zip(docs, scores):
            doc["fine_score"] = float(score)
            doc["final_score"] = (
                    0.3 * doc.get("combined_score", 0) +
                    0.7 * doc["fine_score"]
            )

        return sorted(docs, key=lambda x: x["final_score"], reverse=True)

