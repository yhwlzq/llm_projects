import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from dataclasses import dataclass
from unstructured.partition.pdf import partition_pdf
import networkx as nx
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

# 加载环境变量
load_dotenv()

@dataclass
class DataArgs:
    filename: str = "2024q2-alphabet-earnings-release.pdf"

class HybridRetrieverSystem:
    def __init__(self, config=None):
        # 初始化连接参数
        self.neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "Password123")
        self.config = config or DataArgs()
        
        # 初始化嵌入模型和LLM
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.llm = OllamaLLM(model="deepseek-r1:latest")
        
        # 初始化图数据库连接
        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password
        )
        
        # 初始化图谱转换器
        self.graph_transformer = LLMGraphTransformer(llm=self.llm)

    def load_data(self, filename=None):
        path = Path(__file__).parent
        filename = self.config.filename if filename is None else filename
        filename = os.path.join(path, filename)
        
        # Extract elements from PDF
        raw_pdf_elements = partition_pdf(
            filename=filename,
            extract_images_in_pdf=True,
            strategy="auto",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path="img"
        )
        
        # Process elements
        id_key = 'doc_id'
        documents = []
        for element in raw_pdf_elements:
            if 'unstructured.documents.elements.Table' in str(type(element)):
                documents.append(Document(
                    page_content=str(element),
                    metadata={
                        id_key: str(uuid.uuid4())
                    }
                ))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                documents.append(Document(
                    page_content=str(element),
                    metadata={id_key: str(uuid.uuid4())}
                ))

        return documents

    def extract_and_store_knowledge_graph(self, documents: List[Document]):
        """从文档中提取知识图谱并存储到Neo4j"""
        # 分割文本为适合处理的块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        # 使用LLM提取图谱关系
        graph_documents = self.graph_transformer.convert_to_graph_documents(split_docs)
        
        # 将图谱存储到Neo4j
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        
        return graph_documents

    def visualize_knowledge_graph(self, graph_documents=None, local_visualization=True):
        """可视化知识图谱"""
        if local_visualization:
            # 本地可视化（使用NetworkX和Matplotlib）
            self._visualize_locally()
        else:
            # 从Neo4j获取图谱数据并可视化
            if graph_documents is None:
                graph_documents = self._get_graph_data_from_neo4j()
            
            # 创建NetworkX图
            nx_graph = nx.Graph()
            
            # 添加节点和边
            for doc in graph_documents:
                for node in doc.nodes:
                    nx_graph.add_node(node.id, label=node.type, properties=node.properties)
                for rel in doc.relationships:
                    nx_graph.add_edge(
                        rel.source.node_id,
                        rel.target.node_id,
                        label=rel.type,
                        properties=rel.properties
                    )
            
            # 绘制图谱
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)
            
            # 绘制节点
            nx.draw_networkx_nodes(
                nx_graph, pos,
                node_color='skyblue',
                node_size=500
            )
            
            # 绘制边
            nx.draw_networkx_edges(
                nx_graph, pos,
                edge_color='gray',
                width=1
            )
            
            # 添加标签
            node_labels = {n: f"{nx_graph.nodes[n].get('label', '')}\n{nx_graph.nodes[n].get('properties', {}).get('name', '')}"
                          for n in nx_graph.nodes()}
            nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=8)
            
            edge_labels = {(u, v): nx_graph.edges[u, v]['label']
                          for u, v in nx_graph.edges()}
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=8)
            
            plt.title("Knowledge Graph from PDF")
            plt.axis('off')
            plt.show()

    def _visualize_locally(self):
        """直接从Neo4j数据库可视化图谱"""
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 100
        """
        result = self.graph.query(query)
        
        # 创建NetworkX图
        nx_graph = nx.Graph()
        
        for record in result:
            # 添加节点
            source_node = record['n']
            target_node = record['m']
            relationship = record['r']
            
            nx_graph.add_node(
                source_node.id,
                label=list(source_node.labels)[0],
                properties=dict(source_node)
            )
            
            nx_graph.add_node(
                target_node.id,
                label=list(target_node.labels)[0],
                properties=dict(target_node)
            )
            
            # 添加边
            nx_graph.add_edge(
                source_node.id,
                target_node.id,
                label=relationship.type,
                properties=dict(relationship)
            )
        
        # 绘制图谱
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(
            nx_graph, pos,
            node_color='skyblue',
            node_size=500
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            nx_graph, pos,
            edge_color='gray',
            width=1
        )
        
        # 添加标签
        node_labels = {n: f"{nx_graph.nodes[n].get('label', '')}\n{nx_graph.nodes[n].get('properties', {}).get('name', '')}"
                      for n in nx_graph.nodes()}
        nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=8)
        
        edge_labels = {(u, v): nx_graph.edges[u, v]['label']
                      for u, v in nx_graph.edges()}
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Knowledge Graph from Neo4j")
        plt.axis('off')
        # plt.show()
        plt.savefig('knowledge_graph.png')
        print("知识图谱已保存为 knowledge_graph.png")

    def _get_graph_data_from_neo4j(self):
        """从Neo4j获取图谱数据"""
        query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 100
        """
        result = self.graph.query(query)
        return result

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

    def create_hybrid_retriever(
        self,
        vector_index_name: str = "vector",
        keyword_index_name: str = "keyword",
        graph_weight: float = 0.3,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.2
    ) -> EnsembleRetriever:
        """创建混合检索器"""
        try:
            graph_retriever = self._create_graph_retriever()
            
            # 使用Neo4Vector的as_retriever()方法替代Neo4jVectorRetriever
            vector_retriever = Neo4jVector.from_existing_index(
                embedding=self.embeddings,
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=vector_index_name,
                keyword_index_name=keyword_index_name,
                search_type="hybrid"
            ).as_retriever()
            
            return EnsembleRetriever(
                retrievers=[graph_retriever, vector_retriever],
                weights=[graph_weight, vector_weight + keyword_weight]
            )
        except Exception as e:
            raise RuntimeError(f"创建混合检索器失败: {str(e)}")

    def _create_graph_retriever(self):
        """创建图谱检索器"""
        def graph_query(input: str) -> List[Document]:
            query = """
            MATCH (n)-[r]->(m)
            WHERE n.name CONTAINS $query OR m.name CONTAINS $query
            RETURN n, r, m
            LIMIT 10
            """
            result = self.graph.query(query, {"query": input})
            
            docs = []
            for record in result:
                content = f"{record['n']['name']} -[{record['r'].type}]-> {record['m']['name']}"
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": "graph",
                        "node_id": record['n'].id,
                        "relationship": record['r'].type,
                        "target_id": record['m'].id
                    }
                ))
            return docs
        
        return RunnableLambda(graph_query)

    def query_retriever(
        self,
        retriever: EnsembleRetriever,
        query: str,
        top_k: int = 5
    ) -> List[Document]:
        """使用混合检索器查询"""
        return retriever.invoke(query)[:top_k]

# 使用示例
if __name__ == "__main__":
    try:
        # 1. 初始化系统
        system = HybridRetrieverSystem()
        
        # 2. 加载PDF文档
        documents = system.load_data()
        
        # 3. 从文档中提取知识图谱并存储
        print("正在从PDF提取知识图谱...")
        graph_documents = system.extract_and_store_knowledge_graph(documents)
        
        # 4. 可视化知识图谱
        print("正在可视化知识图谱...")
        system.visualize_knowledge_graph(local_visualization=True)
        
        # 5. 初始化向量存储
        print("正在初始化向量存储...")
        vector_store = system.initialize_vector_store(documents, overwrite=True)
        
        # 6. 创建混合检索器
        print("正在创建混合检索器...")
        hybrid_retriever = system.create_hybrid_retriever()
        
        # 7. 执行查询
        query = "数据集中最常讨论的主题是什么?"
        print(f"\n执行查询: {query}")
        results = system.query_retriever(hybrid_retriever, query)
        
        print("\n混合检索结果:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content}")
            print(f"   来源: {doc.metadata.get('source', '未知')}")
            print(f"   相关度: {doc.metadata.get('score', 0):.2f}")
            print("-" * 50)
            
    except Exception as e:
        print(f"系统错误: {str(e)}")