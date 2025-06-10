from langchain_ollama  import OllamaEmbeddings, OllamaLLM
from langchain_neo4j import Neo4jGraph
from dataclasses import dataclass
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from pathlib import Path
import os
from hashlib import md5
import uuid
from typing import List


@dataclass
class DataArgs:
    file_name_nodes:str = "nodes.csv"
    file_name_relationship:str = "relationships.csv"
    file_name_pdf:str = "rag_graph/2024q2-alphabet-earnings-release.pdf"

    # neo4j configuration
    neo4j_url:str ="neo4j://localhost:7687"
    neo4j_username:str = "neo4j"
    neo4j_password:str = "Password123"

    # model configuration
    llm_model_name_or_path ="deepseek-r1:latest"
    embedding_model_name_or_path = "nomic-embed-text:latest"

    # use to identity doc
    document_hash_property:str = "doc_hash"
    

class HybridRetrieverSystem(object):
    def __init__(self,config=None):
        self.config = config| DataArgs()
        self._init_neo4j()
        self._init_model_and_embbeding()

    
    def _init_neo4j(self):
         sneo4j_url = os.getenv("NEO4J_URI",self.config.neo4j_url)
         neo4j_username = os.getenv("NEO4J_URI",self.config.neo4j_username)
         neo4j_password = os.getenv("NEO4J_URI",self.config.neo4j_password)
         self.graph = Neo4jGraph(url=sneo4j_url,username=neo4j_username,password=neo4j_password)

    def _init_model_and_embbeding(self):
        self.llm = OllamaLLM(model = self.cofnig.model_name_or_path)
        self.embbeding = OllamaEmbeddings(model = self.config.embedding_model_name_or_path )


    def get_label_for_type(node_type):
        mappings ={
             "Supplier": "Supplier",
            "Manufacturer": "Manufacturer",
            "Distributor": "Distributor",
            "Retailer": "Retailer",
            "Product": "Product"
        }
        return mappings.get(node_type,"Entity")
    

    def _generate_document_hash(self,content:str)->str:
        return md5(content.encode("utf-8")).hexdigest()
    
    def _document_exists(self,doc_hash:str)->bool:
        query =f"""
              MATCH (d:Document {{{self.config.document_hash_property}: $doc_hash}})
              RETURN count(d) > 0 AS exists
        """
        result = self.graph.query(query,{"doc_hash":doc_hash})
        return result[0]['exists'] if result else False

    
    def load_data(self,file_name=None)->List[Document]:
        file_name = file_name | self.config.file_name_pdf
        file_path = Path(__file__).parent
        file_name = os.path.join(file_path,file_name)
        raw_pdf_elements  = partition_pdf(
                filename=file_name,
                extract_images_in_pdf=True,
                strategy='auto',
                max_characters=4000,
                extract_image_block_output_dir='img',
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
        )
        documents = []
        for element in raw_pdf_elements :
            element_str = str(element)
            doc_hash = self._generate_document_hash(element_str)
            if not self._document_exists(doc_hash):
                documents.append(Document(
                    page_content=element_str,
                    metadata ={
                        "doc_id":str(uuid.uuid4()),
                        self.config.document_hash_property: doc_hash,
                        "source":file_name
                    }
                ))
        return documents




