import chromadb
from diskcache import Cache
from PIL import Image
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib
import os
from functools import lru_cache
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class VectorDbConfig:
    """Centralized configuration management"""
    vector_db_path: str = "./chroma_db"
    cache_dir: str = "./vector_store"
    image_storage_dir: str = "./pdf_images"
    collection_name: str = "pdf_documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    sparse_model: str = "bm25"  # or "tfidf"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_workers: int = 4
    tesseract_path: Optional[str] = None
    min_image_size: Tuple[int, int] = (100, 100)
    sparse_top_k: int = 1000
    retrieval_top_k: int = 100  # Initial retrieval size before reranking
    rerank_top_k: int = 20  # Number of docs to rerank
    final_top_k: int = 5  # Final number of results to return
    sparse_weight: float = 0.3  # Weight for sparse scores in hybrid retrieval
    dense_weight: float = 0.7  # Weight for dense scores in hybrid retrieval


class VectorStoreManager:

    def __init__(self, config: Optional[VectorDbConfig] = None):
        """Initialize vector store with enhanced retrieval capabilities"""
        self.config = config if config else VectorDbConfig()
        self._ensure_dirs_exist()
        self.client = self._initialize_client()
        self.embedding_fn = self._init_embedding_model()
        self.sparse_model = self._init_sparse_model()
        self.rerank_model = self._init_rerank_model()
        self.cache = Cache(directory=self.config.cache_dir)

    def _ensure_dirs_exist(self):
        """Ensure required directories exist"""
        os.makedirs(self.config.vector_db_path, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.image_storage_dir, exist_ok=True)

    def _initialize_client(self) -> chromadb.Client:
        """Initialize ChromaDB client"""
        return chromadb.PersistentClient(
            path=self.config.vector_db_path,
            settings=Settings(allow_reset=True, anonymized_telemetry=False))

    def _init_embedding_model(self):
        """Initialize dense embedding model"""
        try:
            return HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace model, using default: {str(e)}")
            return embedding_functions.DefaultEmbeddingFunction()


    def _get_collection(self, collection_name: Optional[str] = None) -> chromadb.Collection:
        """安全获取集合，如果不存在则创建"""
        collection_name = collection_name or self.config.collection_name
        try:
            return self.client.get_collection(name=collection_name)
        except Exception as e:
            logger.info(f"Collection {collection_name} not found, creating new one")
            return self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    @lru_cache(maxsize=1000)
    def is_exist(self, doc_id: str, collection_name: Optional[str] = None) -> bool:
        """检查文档是否已存在"""
        try:
            collection = self._get_collection(collection_name)
            existing = collection.get(ids=[doc_id])
            return len(existing["ids"]) > 0
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            return False

    def store_document(self, documents: List[Dict], collection_name: Optional[str] = None) -> bool:
        """
        存储文档到向量数据库
        Args:
            documents: 文档列表，每个文档应包含text和metadata
            collection_name: 集合名称
        Returns:
            bool: 是否存储成功
        """
        if not documents:
            logger.warning("No documents provided for upsert")
            return False
        collection_name = collection_name or self.config.collection_name

        collection = self._get_collection(collection_name)
        batch_size = min(100, getattr(self.embedding_fn, 'encode_kwargs', {}).get('batch_size', 32))

        try:
            # 准备数据
            ids = []
            texts = []
            metadatas = []
            id_set = set()

            for doc in documents:
                if not isinstance(doc, dict) or "text" not in doc or "metadata" not in doc:
                    logger.error("Invalid document format, must contain 'text' and 'metadata'")
                    continue

                # doc_hash = doc["metadata"].get("doc_hash", hashlib.md5(doc["text"].encode()).hexdigest())
                # doc_id = f"doc_{doc_hash[:16]}"
                doc_id = doc["metadata"].get("doc_id")

                # 处理重复ID
                original_id = doc_id
                suffix = 1
                while doc_id in id_set:
                    doc_id = f"{original_id}_{suffix}"
                    suffix += 1

                id_set.add(doc_id)
                ids.append(doc_id)
                texts.append(doc["text"])
                metadatas.append(doc["metadata"])

            # 批量处理嵌入和存储
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]

                try:
                    embeddings = self.embedding_fn.embed_documents(batch_texts)
                    collection.upsert(
                        ids=batch_ids,
                        embeddings=embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadatas
                    )
                    logger.debug(f"Upserted batch {i // batch_size + 1}")
                except Exception as e:
                    logger.error(f"Failed to process batch {i // batch_size + 1}: {str(e)}")
                    continue

            logger.info(f"Successfully stored {len(ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}", exc_info=True)
            return False

    def _init_sparse_model(self):
        """Initialize sparse retrieval model (BM25 or TF-IDF)"""
        if self.config.sparse_model == "bm25":
            return None  # Will be initialized with corpus data when needed
        elif self.config.sparse_model == "tfidf":
            return TfidfVectorizer(analyzer='word', stop_words='english')
        else:
            raise ValueError(f"Unsupported sparse model: {self.config.sparse_model}")

    def _init_rerank_model(self):
        """Initialize reranking model"""
        try:
            return CrossEncoder(self.config.rerank_model, max_length=512)
        except Exception as e:
            logger.warning(f"Failed to load rerank model: {str(e)}")
            return None

    def _get_collection(self, collection_name: Optional[str] = None) -> chromadb.Collection:
        """Safely get collection, create if not exists"""
        collection_name = collection_name or self.config.collection_name
        try:
            return self.client.get_collection(name=collection_name)
        except Exception as e:
            logger.info(f"Collection {collection_name} not found, creating new one")
            return self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def _initialize_sparse_index(self, documents: List[str]) -> Any:
        """Initialize sparse index with document corpus"""
        if self.config.sparse_model == "bm25":
            tokenized_docs = [doc.split() for doc in documents]
            return BM25Okapi(tokenized_docs)
        elif self.config.sparse_model == "tfidf":
            return self.sparse_model.fit_transform(documents)
        return None

    def _sparse_retrieve(self, query: str, documents: List[str], sparse_index: Any) -> List[Tuple[int, float]]:
        """Perform sparse retrieval"""
        if self.config.sparse_model == "bm25":
            tokenized_query = query.split()
            scores = sparse_index.get_scores(tokenized_query)
            return [(i, float(score)) for i, score in enumerate(scores)]
        elif self.config.sparse_model == "tfidf":
            query_vec = self.sparse_model.transform([query])
            scores = (sparse_index * query_vec.T).toarray().flatten()
            return [(i, float(score)) for i, score in enumerate(scores)]
        return []

    def _dense_retrieve(self, query_embedding: List[float], collection: chromadb.Collection) -> List[Dict]:
        """Perform dense retrieval using vector similarity"""
        retrieved = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.config.retrieval_top_k,
            include=["documents", "metadatas", "distances"]
        )

        results = []
        for i in range(len(retrieved["ids"][0])):
            results.append({
                "id": retrieved["ids"][0][i],
                "text": retrieved["documents"][0][i],
                "metadata": retrieved["metadatas"][0][i],
                "dense_score": 1.0 - retrieved["distances"][0][i],
                "sparse_score": 0.0,
                "combined_score": 0.0
            })
        return results

    def _hybrid_retrieve(self, query: str, query_embedding: List[float], collection: chromadb.Collection) -> List[Dict]:
        """Perform hybrid sparse-dense retrieval"""
        # First get all documents for sparse retrieval
        all_docs = collection.get(include=["documents"])
        documents = all_docs["documents"]

        # Perform sparse retrieval
        sparse_index = self._initialize_sparse_index(documents)
        sparse_scores = self._sparse_retrieve(query, documents, sparse_index)

        # Convert sparse scores to dictionary for easy lookup
        sparse_score_dict = {doc_id: score for doc_id, score in sparse_scores}

        # Perform dense retrieval
        dense_results = self._dense_retrieve(query_embedding, collection)

        # Combine scores
        for result in dense_results:
            doc_idx = documents.index(result["text"])
            result["sparse_score"] = sparse_score_dict.get(doc_idx, 0.0)
            # Normalize scores
            result["combined_score"] = (
                    self.config.sparse_weight * result["sparse_score"] +
                    self.config.dense_weight * result["dense_score"]
            )

        return sorted(dense_results, key=lambda x: x["combined_score"], reverse=True)

    def _coarse_rerank(self, results: List[Dict]) -> List[Dict]:
        """First stage reranking with simple heuristics"""
        if not results:
            return results

        # Apply basic filters
        filtered = []
        for result in results:
            # Example: Filter out very short documents
            if len(result["text"]) < 50:
                continue
            filtered.append(result)

        # Take top k for fine reranking
        return filtered[:self.config.rerank_top_k]

    def _fine_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Second stage reranking with cross-encoder"""
        if not self.rerank_model or not results:
            return results[:self.config.final_top_k]

        # Prepare query-doc pairs for reranking
        query_doc_pairs = [(query, result["text"]) for result in results]

        # Get rerank scores
        rerank_scores = self.rerank_model.predict(query_doc_pairs)

        # Update results with rerank scores
        for i, score in enumerate(rerank_scores):
            results[i]["rerank_score"] = float(score)

        # Sort by rerank score
        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:self.config.final_top_k]

    # ----------- 核心检索流程 -----------
    def retrieve(
            self,
            query: Union[str, Image.Image],
            top_k: Optional[int] = None,
            score_threshold: Optional[float] = None,
            collection_name: Optional[str] = None,
            **kwargs
    ) -> List[Dict]:
        """
        完整检索流程：
        1. 稀疏召回 -> 2. 稠密召回 -> 3. 粗排 -> 4. 精排
        """
        # 参数处理
        final_top_k = top_k or self.config.final_top_k
        collection_name = collection_name or self.config.collection_name

        # 缓存检查
        cache_key = self._generate_cache_key(query, final_top_k, score_threshold, collection_name, kwargs)
        if cached := self.cache.get(cache_key):
            return cached

        try:
            # 阶段0：准备查询
            query_text, query_embedding = self._preprocess_query(query)
            collection = self._get_collection(collection_name)

            # 阶段1：稀疏召回
            sparse_results = self._sparse_retrieval(
                query_text,
                collection,
                top_k=self.config.sparse_top_k
            )

            # 阶段2：稠密召回
            dense_results = self._dense_retrieval(
                query_embedding,
                collection,
                candidate_ids=[r["id"] for r in sparse_results],
                top_k=self.config.dense_top_k
            )

            # 阶段3：粗排
            coarse_results = self._coarse_ranking(
                query_text,
                dense_results,
                top_k=self.config.coarse_top_k
            )

            # 阶段4：精排
            final_results = self._fine_ranking(
                query_text,
                coarse_results,
                top_k=final_top_k,
                score_threshold=score_threshold
            )

            # 缓存结果
            self.cache.set(cache_key, final_results, expire=3600)
            return final_results

        except Exception as e:
            logger.error(f"Retrieval pipeline failed: {str(e)}", exc_info=True)
            return []

    # ----------- 各阶段具体实现 -----------
    def _sparse_retrieval(
            self,
            query: str,
            collection: chromadb.Collection,
            top_k: int
    ) -> List[Dict]:
        """阶段1：稀疏召回（BM25/TF-IDF）"""
        # 获取全部文档语料
        if not self.cache[collection.name].get("corpus"):
            all_docs = collection.get(include=["documents"])
            self.cache[collection.name]["corpus"] = all_docs["documents"]
            self.cache[collection.name]["doc_ids"] = all_docs["ids"]

        corpus = self.cache[collection.name]["corpus"]
        doc_ids = self.cache[collection.name]["doc_ids"]

        # 初始化稀疏模型
        if self.config.sparse_model_type == "bm25":
            if not self.sparse_model:
                tokenized_corpus = [doc.split() for doc in corpus]
                self.sparse_model = BM25Okapi(tokenized_corpus)
            scores = self.sparse_model.get_scores(query.split())
        else:  # TF-IDF
            if not self.sparse_model:
                self.sparse_model = TfidfVectorizer().fit(corpus)
            query_vec = self.sparse_model.transform([query])
            doc_vecs = self.sparse_model.transform(corpus)
            scores = (doc_vecs * query_vec.T).toarray().flatten()

        # 获取TopK结果
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [{
            "id": doc_ids[i],
            "text": corpus[i],
            "sparse_score": float(scores[i]),
            "metadata": collection.get(ids=[doc_ids[i]], include=["metadatas"])["metadatas"][0]
        } for i in top_indices if scores[i] > 0]

    def _dense_retrieval(
            self,
            query_embedding: List[float],
            collection: chromadb.Collection,
            candidate_ids: List[str],
            top_k: int
    ) -> List[Dict]:
        """阶段2：稠密召回（向量检索）"""
        # 在稀疏召回的结果基础上做稠密检索
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"id": {"$in": candidate_ids}} if candidate_ids else None,
            include=["documents", "metadatas", "distances"]
        )

        return [{
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "dense_score": 1.0 - results["distances"][0][i],
            "combined_score": 0.0  # 将在粗排阶段计算
        } for i in range(len(results["ids"][0]))]

    def _coarse_ranking(
            self,
            query: str,
            candidates: List[Dict],
            top_k: int
    ) -> List[Dict]:
        """阶段3：粗排（混合分数+启发式规则）"""
        if not candidates:
            return []

        # 1. 混合分数计算
        max_sparse = max(r.get("sparse_score", 0) for r in candidates) or 1
        max_dense = max(r.get("dense_score", 0) for r in candidates) or 1

        for r in candidates:
            # 分数归一化
            norm_sparse = r.get("sparse_score", 0) / max_sparse
            norm_dense = r.get("dense_score", 0) / max_dense
            # 加权混合
            r["combined_score"] = (
                    self.config.hybrid_weights[0] * norm_sparse +
                    self.config.hybrid_weights[1] * norm_dense
            )

        # 2. 启发式过滤
        filtered = [
            r for r in candidates
            if self.config.min_doc_length <= len(r["text"]) <= self.config.max_doc_length
        ]

        # 3. 按混合分数排序
        return sorted(filtered, key=lambda x: x["combined_score"], reverse=True)[:top_k]

    def _fine_ranking(
            self,
            query: str,
            candidates: List[Dict],
            top_k: int,
            score_threshold: Optional[float]
    ) -> List[Dict]:
        """阶段4：精排（交叉编码器）"""
        if not candidates or not self.rerank_model:
            return candidates[:top_k]

        # 准备精排数据
        query_doc_pairs = [(query, doc["text"]) for doc in candidates]

        # 批量精排
        rerank_scores = self.rerank_model.predict(query_doc_pairs, batch_size=32)

        # 更新结果
        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)

        # 筛选和排序
        sorted_results = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", x["combined_score"]),
            reverse=True
        )

        # 应用阈值
        if score_threshold is not None:
            sorted_results = [
                r for r in sorted_results
                if r.get("rerank_score", r["combined_score"]) >= score_threshold
            ]

        # 返回精简后的结果
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "score": r.get("rerank_score", r["combined_score"]),
                "details": {
                    "sparse_score": r.get("sparse_score"),
                    "dense_score": r.get("dense_score"),
                    "combined_score": r.get("combined_score")
                }
            }
            for r in sorted_results[:top_k]
        ]

    def _preprocess_query(self, query: Union[str, Image.Image]) -> Tuple[str, List[float]]:
        """查询预处理"""
        if isinstance(query, str):
            return query, self.embedding_fn.embed_query(query)
        elif isinstance(query, Image.Image):
            text = self._image_to_text(query)
            return text, self.embedding_fn.embed_query(text)
        else:
            raise ValueError("Unsupported query type")

    def _image_to_text(self, image: Image.Image) -> str:
        """Convert image to text using OCR"""
        try:
            import pytesseract
            if self.config.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_path
            return pytesseract.image_to_string(image)
        except Exception as e:
            logger.warning(f"OCR failed: {str(e)}")
            return ""

    def _embed_image(self, image: Image.Image) -> List[float]:
        """Embed image using visual model"""
        try:
            from transformers import ViTFeatureExtractor, ViTModel
            import torch

            model_name = "google/vit-base-patch16-224-in21k"
            feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            model = ViTModel.from_pretrained(model_name)

            inputs = feature_extractor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        except Exception as e:
            logger.error(f"Failed to embed image: {str(e)}")
            return [float(hashlib.md5(image.tobytes()).hexdigest()[:16]) / 1e32]

    def _generate_cache_key(self, query: Any, top_k: int, score_threshold: Optional[float],
                            collection_name: str,  kwargs: Dict) -> str:
        """Generate cache key"""
        if isinstance(query, str):
            query_key = hashlib.md5(query.encode()).hexdigest()
        elif isinstance(query, Image.Image):
            query_key = hashlib.md5(query.tobytes()).hexdigest()
        else:
            query_key = str(hash(query))

        params = f"{top_k}_{score_threshold}_{collection_name}_{str(kwargs)}"
        return f"{query_key}_{hashlib.md5(params.encode()).hexdigest()}"


print(VectorStoreManager().retrieve("Deep learning is a subset of machine learning that focuses on training artificial neural networks to perform "))