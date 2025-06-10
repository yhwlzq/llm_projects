import logging

import chromadb
from diskcache import Cache
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions
import hashlib
from settings import settings, FileType
from typing import Dict, Set, List, Union
from functools import lru_cache

logger = logging.getLogger(__name__)

class VectorManager(object):

    DEFAULT_FILE_TYPES = list(FileType)  #

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR)
        )
        self.cache = self._init_cache()
        self.embedding_fn = self._init_embbeding()
        self._init_collection_hashes()
        pass

    def _init_cache(self):
        return Cache(
            directory=str(settings.CACHE_DIR / "vector_store"),
            size_limit=1_000_000_000,  # 1GB
            eviction_policy='least-recently-used'
        )

    def _init_embbeding(self):
        try:
            return HuggingFaceEmbeddings(
                mode_name = settings.EMBEDDING_MODEL,
                model_kwargs = {'device':'cpu'},
                encode_kwards = {'normalize_embeddings':True}
            )
        except Exception as e:
            return embedding_functions.DefaultEmbeddingFunction()

    def _init_collection_hashes(self):
        self.existing_hashes:Dict[str:Set[str]] ={
            ft.value:set() for ft in FileType
        }
        for col in self.client.list_collections():
            for doc in col.get()['metadatas']:
                if 'doc_hash' in doc:
                    self.existing_hashes[col.name].add(doc['doc_hash'])

    def _generate_cache_key(self,query:str, file_types:List[Union[FileType, str]], top_k:int) ->str:
        normalized_type = ",".join(sorted(ft.value if isinstance(ft, FileType)  else ft for ft in file_types))
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"retrieve__{query_hash}_{normalized_type}_{top_k}"

    @lru_cache(maxsize=128)
    def _get_collection_name(self,file_type:Union[FileType,str])->str:
        prefix = settings.COLLECTION_PREFIX
        return  f"{prefix}_{file_type.value if isinstance(file_type, FileType) else file_type}"

    def _invalidate_cache_for_filetype(self, file_type:Union[FileType,str]):
        if isinstance(file_type, FileType):
            file_type = file_type.value
        for key in list(self.cache.iterkeys()):
            if f"filetype_{file_type}_" in key:
                del self.cache[key]
                logger.debug(f"Invalidated cache key: {key}")

    def _execute_query(self,query:str,  file_types: List[Union[FileType, str]], top_k: int) ->List[Dict]:
        try:
            query_embedding = self.embedding_fn.embed_query(query)
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            return []
        results = []
        for ft in file_types:
            col_name = self._get_collection_name(ft)
            if col_name not in [c.name for c in self.client.list_collections()]:
                continue
            try:
                col = self.client.get_collection(col_name)
                retrieved = col.query(
                    query_embeddings=query_embedding,
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )

                for i in range(len(retrieved['id'][0])):
                    results.append({
                        "text": retrieved["documents"][0][i],
                        "metadata": retrieved["metadatas"][0][i],
                        "score": retrieved["distances"][0][i],
                        "file_type": ft.value if isinstance(ft, FileType) else ft
                    })
            except Exception as e:
                logger.warning(f"Query failed on {col_name}: {str(e)}")
        return sorted(results,key=lambda x:x['score'])[:top_k]


    def retrieve(self, query:str, file_types:List[Union[FileType,str]]=DEFAULT_FILE_TYPES,top_k:int=5):
        cache_key = self._generate_cache_key(query,file_types,top_k)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for: {query[:30]}...")
            return self.cache[cache_key]
        pass




