from typing import Dict, Generator, Union
from pathlib import Path
import logging
from csv_processor import CSVProcessor
from pdf_processor import PDFProcessor
from settings import settings, FileType

logger = logging.getLogger(__name__)

class DocumentProcessor(object):
    def __init__(self):
        self.processor = {
            FileType.CSV.value:CSVProcessor(),
            FileType.PDF.value:PDFProcessor()
        }

    def process(self, file_path:Path, file_type:Union[FileType,str])->Generator[Dict,None,None]:
        if isinstance(file_type, FileType):
            file_type = file_type.value
        if file_type not in self.processor:
            raise ValueError(f"Unsupported file type:{file_type}")

        processor = self.processor[file_type]
        yield from processor.process(file_path)





































