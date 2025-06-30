from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dataclasses import dataclass
from unstructured.partition.pdf import partition_pdf
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pathlib import Path
import os
from hashlib import md5
import uuid
from typing import List
import logging
import tiktoken

logging.getLogger("pdfminer").setLevel(logging.ERROR)  # 屏蔽pdfminer警告
logging.getLogger("pymupdf").setLevel(logging.ERROR)  # 屏蔽pymupdf警告

logger = logging.getLogger(__name__)


@dataclass
class DataArgs:
    file_name_nodes: str = "nodes.csv"
    file_name_relationship: str = "relationships.csv"
    file_name_pdf: str = "2024q2-alphabet-earnings-release.pdf"

    # neo4j configuration
    neo4j_url: str = "neo4j://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "Password123"

    # model configuration
    llm_model_name_or_path = "deepseek-r1:latest"
    embedding_model_name_or_path = "nomic-embed-text:latest"
    rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # use to identity doc
    document_hash_property: str = "doc_hash"


class HybridSystem:
    def __init__(self, config=None):
        self.config = config or DataArgs()
        self._init_neo4j()
        self._init_model_and_embbeding()

    def _init_neo4j(self):
        self.sneo4j_url = os.getenv("NEO4J_URI", self.config.neo4j_url)
        self.neo4j_username = os.getenv("NEO4J_URI", self.config.neo4j_username)
        self.neo4j_password = os.getenv("NEO4J_URI", self.config.neo4j_password)
        self.graph = Neo4jGraph(url=self.sneo4j_url, username=self.neo4j_username, password=self.neo4j_password)

    def _init_model_and_embbeding(self):
        self.llm = OllamaLLM(model=self.config.llm_model_name_or_path)
        self.embbeding = OllamaEmbeddings(model=self.config.embedding_model_name_or_path)
        self.graph_llm = LLMGraphTransformer(llm=self.llm)


class HybridRetrieverDocSystem(HybridSystem):
    def __init__(self, config=None):
        super().__init__(config)
        self._initialize_database_schema()

    def _initialize_database_schema(self):
        """确保属性和约束都存在"""
        # 1. 先创建一个带该属性的临时节点（如果属性不存在会自动创建）
        init_query = f"""
            MERGE (d:Document {{ {self.config.document_hash_property}: "TEMP_INIT_VALUE" }})
            DELETE d
        """

        # 2. 然后创建唯一约束
        constraint_query = f"""
            CREATE CONSTRAINT IF NOT EXISTS 
            FOR (d:Document) REQUIRE d.{self.config.document_hash_property} IS UNIQUE
        """

        try:
            self.graph.query(init_query)  # 先确保属性存在
            self.graph.query(constraint_query)  # 再创建约束
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def get_label_for_type(node_type):
        mappings = {
            "Supplier": "Supplier",
            "Manufacturer": "Manufacturer",
            "Distributor": "Distributor",
            "Retailer": "Retailer",
            "Product": "Product"
        }
        return mappings.get(node_type, "Entity")

    def _generate_document_hash(self, content: str) -> str:
        return md5(content.encode("utf-8")).hexdigest()

    def _document_exists(self, doc_hash: str) -> bool:
        query = f"""
              MATCH (d:Document {{{self.config.document_hash_property}: $doc_hash}})
              RETURN count(d) > 0 AS exists
        """
        result = self.graph.query(query, {"doc_hash": doc_hash})
        return result[0]['exists'] if result else False

    def store(self, file_name=None, is_overwrite=False):
        file_name = file_name or self.config.file_name_pdf
        file_path = Path(__file__).parent
        file_name = os.path.join(file_path, file_name)

        # step 1: fetch document
        documents = self.chunk(file_name)

        # step 2:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = splitter.split_documents(documents)

        # step 3:
        graph_docs = self.graph_llm.aconvert_to_graph_documents(split_docs)
        self.graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

        # step 4:
        self.initialize_vector_store(file_name, is_overwrite)

    def chunk(self, file_name=None) -> List[Document]:
        raw_pdf_elements = partition_pdf(
            filename=file_name,
            extract_images_in_pdf=True,
            strategy='auto',
            max_characters=4000,
            extract_image_block_output_dir='img',
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
        )
        documents = []
        for element in raw_pdf_elements:
            element_str = str(element)
            doc_hash = self._generate_document_hash(element_str)
            if not self._document_exists(doc_hash):
                documents.append(Document(
                    page_content=element_str,
                    metadata={
                        "doc_id": str(uuid.uuid4()),
                        self.config.document_hash_property: doc_hash,
                        "source": file_name
                    }
                ))
        return documents

    def initialize_vector_store(
            self,
            documents: List[Document],
            index_name: str = "vector",
            keyword_index_name: str = "keyword",
            overwrite: bool = True
    ) -> Neo4jVector:
        """初始化Neo4j向量存储"""
        try:
            if overwrite:
                self._clean_existing_indices(index_name, keyword_index_name)

            return Neo4jVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=index_name,
                keyword_index_name=keyword_index_name,
                search_type="hybrid"
            )
        except Exception as e:
            raise RuntimeError(f"初始化向量存储失败: {str(e)}")

    def _clean_existing_indices(self, *index_names):
        """清理现有索引"""
        for index_name in index_names:
            self.graph.query(f"DROP INDEX {index_name} IF EXISTS")


class HybridRetrieverWithRerank(HybridSystem):
    def __init__(self, config=None):
        super().__init__(config)
        self._init_retriever()
        self._init_reranker()

    def _init_retriever(self):
        vector_retriever = Neo4jVector.from_existing_index(
            embedding=self.embbeding,
            url=self.sneo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            index_name="vector",
            keyword_index_name="keyword",
            search_type="hybrid"  # 启用混合搜索
        ).as_retriever(search_kwargs={"k": 20})

        graph_retrever = self._create_graph_retriever()
        self.retriever = EnsembleRetriever(retrievers=[vector_retriever, graph_retrever], weights=[0.6, 0.4])

    def _create_graph_retriever(self) -> BaseRetriever:
        class GraphRetriever(BaseRetriever):
            graph: Neo4jGraph = None

            def __init__(self, graph: Neo4jGraph):
                super().__init__(graph=graph)
                self.graph = graph

            def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                cypher_query = """
                    MATCH (doc:Document)-[r]->(entity)
                    WHERE doc.text CONTAINS $query OR entity.name CONTAINS $query
                    RETURN doc.text AS text, properties(doc) AS metadata
                    LIMIT 10
                    """
                result = self.graph.query(cypher_query, {"query": query})
                return [
                    Document(
                        page_content=record["text"],
                        metadata=record["metadata"]
                    ) for record in result
                ]

            async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                return self._get_relevant_documents(query, **kwargs)
        return GraphRetriever(self.graph)

    def _init_reranker(self):
       self.rerank_model = CrossEncoder(model_name_or_path=self.config.rerank_model)

    async def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.rerank_model.predict(pairs)
        for doc, score in zip(documents, scores):
            doc.metadata['score'] = float(score)
        return sorted(documents,key=lambda x:x.metadata['score'],reverse=True)

    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        recalled_docs = await self.retriever.ainvoke(query)
        unique_docs = []
        seen_ids = set()
        for doc in recalled_docs:
            if not isinstance(doc, Document):
                continue
            doc_id = doc.metadata.get("doc_id", str(hash(doc.page_content)))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        reranked_docs = await self.rerank_documents(query, unique_docs)
        return reranked_docs[:top_k]

async def main():
    query = "how about Alphabet's Financial Performance?"
    print(f"\n执行查询: {query}")
    results = await HybridRetrieverWithRerank().retrieve(query)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")
        print(f"   来源: {doc.metadata.get('source', '未知')}")
        print(f"   相关度: {doc.metadata.get('score', 0):.2f}")
        print("-" * 50)
    print("\n混合检索结果:")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


