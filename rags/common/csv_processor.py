import hashlib
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Generator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from settings import settings
import json


class CSVProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CSV_CHUNK_SIZE,
            chunk_overlap=settings.TEXT_CHUNK_OVERLAB
        )

    def process(self, file_path:Path) ->Generator[Dict[str,any],None,None]:
        try:
            df = pd.read_csv(file_path)
            for row_idx, row in df.iterrows():
                row_data = row.to_dict()
                row_hash = hashlib.md5(f"{file_path.name}_{row_idx}".encode()).hexdigest()
                metadata = {
                    'source':file_path.name,
                    'row_id':str(row_idx),
                    "doc_hash":row_hash
                }

                structured_content = []
                for col_name, value in row_data.items():
                    if pd.isna(value):
                        continue
                    structured_content.append(f"{col_name}:{value}")
                full_text = '\n'.join(structured_content)
                chunks = self.text_splitter.split_text(full_text)
                for chunk_idx,chunk in enumerate(chunks):
                    yield {
                        'text':chunk,
                        'metadata':{
                            **metadata,
                            'chunk_count':len(chunk),
                            "chunk_id":str(chunk_idx),
                            "full_row_json":json.dumps(row_data,ensure_ascii=False)
                        }
                    }
        except Exception as e:
            self.logger.error(f"Failed to process CSV file {file_path}: {str(e)}")
            raise

