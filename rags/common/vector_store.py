import chromadb
from diskcache import Cache
from PIL import Image
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions
import logging
from typing import Dict, List, Union, Optional
from settings import settings, FileType
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

# use thread executor to improve performance, currently due to my local, so it is not implement

class VectorStore:
    def __init__(self):
        """初始化向量存储"""
        self.client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=Settings(allow_reset=False)
        )
        self.embedding_fn = self._init_embedding_model()
        self.cache = Cache(directory=str(settings.CACHE_DIR / "vector_store"))

    def _init_embedding_model(self):
        try:
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},  # 安全考虑使用CPU
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace model, using default: {str(e)}")
            return embedding_functions.DefaultEmbeddingFunction()

    @lru_cache(maxsize=128)
    def _get_collection_name(self, file_type: Union[FileType, str]) -> str:
        """获取集合名称"""
        if isinstance(file_type, FileType):
            file_type = file_type.value
        return f"{settings.COLLECTION_PREFIX}_{file_type}"

    def _get_existing_doc_hashes(self, collection_name: str) -> set[str]:
        """获取集合中已存在的文档哈希"""
        try:
            collection = self.client.get_collection(collection_name)
            # 只获取metadata中的doc_hash，避免加载全部数据
            metadatas = collection.get(include=["metadatas"])['metadatas']
            return {m["doc_hash"] for m in metadatas if "doc_hash" in m}
        except Exception as e:
            logger.debug(f"Collection {collection_name} not found or error: {str(e)}")
            return set()

    def _generate_doc_id(self, doc_hash: str, file_type: str, text: str) -> str:
        """生成真正唯一的文档ID"""
        # 使用文档哈希和文本前100个字符的哈希组合，确保唯一性
        text_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
        return f"{file_type}_{doc_hash}_{text_hash}"

    def upsert_documents(self, file_type: Union[FileType, str], documents: List[Dict]) -> bool:
        """
        插入或更新文档，自动去重

        Args:
            file_type: 文件类型 (FileType 枚举或字符串)
            documents: 文档列表，每个文档应包含:
                - text: 文本内容
                - metadata: 元数据字典 (必须包含 doc_hash)

        Returns:
            bool: 操作是否成功
        """
        if not documents:
            logger.warning("No documents provided for upsert")
            return False

        collection_name = self._get_collection_name(file_type)
        file_type_str = file_type.value if isinstance(file_type, FileType) else file_type

        try:
            # 验证所有文档都有 doc_hash
            for doc in documents:
                if "metadata" not in doc or "doc_hash" not in doc["metadata"]:
                    logger.error("Document missing metadata or doc_hash")
                    return False

            # 准备批量数据
            ids = []
            texts = []
            metadatas = []
            id_set = set()  # 用于检测当前批次的重复ID

            for doc in documents:
                doc_hash = doc["metadata"]["doc_hash"]
                text = doc["text"]
                doc_id = self._generate_doc_id(doc_hash, file_type_str, text)

                # 检查当前批次是否有重复ID
                if doc_id in id_set:
                    logger.warning(f"Duplicate ID detected in current batch: {doc_id}")
                    # 为重复ID添加后缀
                    suffix = 1
                    while f"{doc_id}_{suffix}" in id_set:
                        suffix += 1
                    doc_id = f"{doc_id}_{suffix}"

                id_set.add(doc_id)
                ids.append(doc_id)
                texts.append(text)
                metadatas.append(doc["metadata"])

            # 获取或创建集合
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            existing_ids = set()
            if ids:
                try:
                    existing_records = collection.get(ids=ids, include=[])
                    existing_ids = set(existing_records["ids"])
                except Exception as e:
                    logger.debug(f"No existing documents found or error: {str(e)}")

            new_data = {
                "ids": [],
                "texts": [],
                "metadatas": []
            }

            for i, doc_id in enumerate(ids):
                if doc_id not in existing_ids:
                    new_data["ids"].append(doc_id)
                    new_data["texts"].append(texts[i])
                    new_data["metadatas"].append(metadatas[i])

            if not new_data["ids"]:
                logger.debug(f"No new documents to upsert for {collection_name}")
                return True

            batch_size = min(100, self.embedding_fn.encode_kwargs.get('batch_size', 32))

            for i in range(0, len(new_data["texts"]), batch_size):
                batch_texts = new_data["texts"][i:i + batch_size]

                try:
                    embeddings = self.embedding_fn.embed_documents(batch_texts)
                except Exception as e:
                    logger.error(f"Failed to embed documents batch: {str(e)}")
                    continue

                collection.upsert(
                    ids=new_data["ids"][i:i + batch_size],
                    embeddings=embeddings,
                    documents=batch_texts,
                    metadatas=new_data["metadatas"][i:i + batch_size]
                )

            logger.info(f"Upserted {len(new_data['ids'])} new documents to {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert documents to {collection_name}: {str(e)}", exc_info=True)
            return False

    def retrieve(
            self,
            query: str,
            file_types: List[Union[FileType, str]],
            top_k: int = 5,
            score_threshold: Optional[float] = None
    ) -> List[Dict]:
        results = []

        try:
            if isinstance(query, str):
                query_embedding = self.embedding_fn.embed_query(query)
            elif isinstance(query,Image.Image):
                query_embedding = self.model_manager.image_embedder.encode(query)

        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            return []

        for file_type in file_types:
            collection_name = self._get_collection_name(file_type)
            file_type_str = file_type.value if isinstance(file_type, FileType) else file_type

            try:
                collection = self.client.get_collection(collection_name)

                # 使用query参数过滤低质量结果
                retrieved = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )

                for i in range(len(retrieved["ids"][0])):
                    # 将距离转换为相似度 (1-distance)
                    similarity = 1.0 - retrieved["distances"][0][i]

                    if score_threshold is not None and similarity < score_threshold:
                        continue

                    results.append({
                        "text": retrieved["documents"][0][i],
                        "metadata": retrieved["metadatas"][0][i],
                        "score": similarity,  # 使用相似度而不是距离
                        "file_type": file_type_str
                    })

            except chromadb.exceptions.CollectionNotFoundException:
                logger.debug(f"Collection {collection_name} not found, skipping")
            except Exception as e:
                logger.warning(f"Error querying collection {collection_name}: {str(e)}")

        # 按相似度降序排序
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

