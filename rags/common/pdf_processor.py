import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, Generator, List, Any, Optional
from processor.settings import settings
import fitz  # PyMuPDF


class PDFProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.text_chunk_size = settings.TEXT_CHUNK_OVERLAB
        self.table_chunk_size = settings.TABLE_CHUNK_SIZE
        self.extract_images = settings.EXTRACT_IMAGE

    def process(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """
        Process a PDF file and yield chunks for RAG ingestion.

        Args:
            file_path: Path to the PDF file

        Yields:
            Dictionary containing content chunk and metadata

        Raises:
            ValueError: If file doesn't exist or isn't a PDF
            Exception: For other processing errors
        """
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            raise ValueError(f"File not found: {file_path}")

        if file_path.suffix.lower() != '.pdf':
            self.logger.error(f"Not a PDF file: {file_path}")
            raise ValueError(f"Not a PDF file: {file_path}")

        try:
            doc_hash = self._calculate_file_hash(file_path)
            contents = self._extract_content(file_path)

            for item in contents:
                if item['type'] == 'text':
                    yield from self._process_text(item, doc_hash, file_path.name)
                elif item['type'] == "table":
                    yield from self._process_table(item, doc_hash, file_path.name)
                elif item["type"] == "image" and self.extract_images:
                    # yield self._process_image(item, doc_hash, file_path.name)
                    continue
        except Exception as e:
            self.logger.error(f"Failed to process PDF file {file_path}: {str(e)}")
            raise

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for unique identification"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _process_text(self, item: Dict, doc_hash: str, source: str) -> Generator[Dict[str, Any], None, None]:
        """Process text content into appropriate chunks"""
        paragraphs = [p for p in item['content'].split("\n") if p.strip()]
        current_chunk = []
        current_length = 0
        chunk_count = 0

        for para in paragraphs:
            if current_length + len(para) > self.text_chunk_size and current_chunk:
                chunk_id = f"text_{item['page']}_{chunk_count}"
                yield {
                    "text": "\n".join(current_chunk),
                    "metadata": {
                        "type": "text",
                        "page": item["page"],
                        "chunk_id": chunk_id,
                        "doc_hash": doc_hash,
                        "source": source
                    }
                }
                current_chunk = []
                current_length = 0
                chunk_count += 1

            current_chunk.append(para)
            current_length += len(para)

        if current_chunk:
            chunk_id = f"text_{item['page']}_{chunk_count}"
            yield {
                "text": "\n".join(current_chunk),
                "metadata": {
                    "type": "text",
                    "page": item["page"],
                    "chunk_id": chunk_id,
                    "doc_hash": doc_hash,
                    "source": source
                }
            }

    def _process_table(self, item: Dict, doc_hash: str, source: str) -> Generator[Dict[str, Any], None, None]:
        """Process table content into appropriate chunks"""
        rows = item['content'].split("\n")
        for i in range(0, len(rows), self.table_chunk_size):
            chunk_rows = rows[i:i + self.table_chunk_size]
            chunk_id = f"table_{item['page']}_{i // self.table_chunk_size}"

            valid_rows = [row for row in chunk_rows if row.strip()]
            if not valid_rows:continue
            yield {
                "text": "\n".join(chunk_rows),
                "metadata": {
                    "type": "table",
                    "page": item['page'],
                    "chunk_id": chunk_id,
                    "doc_hash": doc_hash,
                    "source": source,
                    "row_start": i,
                    "row_end": min(i + self.table_chunk_size, len(rows))
                }
            }

    def _process_image(self, item: Dict, doc_hash: str, source: str) -> Dict[str, Any]:
        """Process image content"""
        return {
            "image": item["content"],
            "metadata": {
                "type": "image",
                "page": item['page'],
                "chunk_id": f"image_{item['page']}_{item.get('img_index', 0)}",
                "doc_hash": doc_hash,
                "source": source,
                "format": item.get("format", "unknown")
            }
        }

    def _extract_content(self, file_path: Path) -> List[Dict]:
        """Extract all content from PDF including text, tables and images"""
        try:
            doc = fitz.open(file_path)
            num_pages = doc.page_count
            self.logger.info(f"Processing PDF with {num_pages} pages: {file_path.name}")

            chunks = []
            for page_num in range(num_pages):
                page = doc.load_page(page_num)

                # Extract text
                text = page.get_text("text")
                if text.strip():
                    chunks.append({
                        "type": "text",
                        "content": text,
                        "page": page_num + 1
                    })

                # Extract tables
                try:
                    tables = page.find_tables()
                    if tables and tables.tables:
                        for table_idx, table in enumerate(tables.tables):
                            csv_data = self._safe_extract_table(table)
                            if csv_data.strip():
                                chunks.append({
                                    "type": "table",
                                    "content": csv_data,
                                    "page": page_num + 1,
                                    "table_idx": table_idx
                                })
                except Exception as e:
                    self.logger.warning(f"Failed to extract tables from page {page_num + 1}: {str(e)}")

                # Extract images
                if self.extract_images:
                    try:
                        for img_index, img in enumerate(page.get_images(full=True)):
                            xref = img[0]
                            try:
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image['image']

                                # Base64 encode the image
                                img_b64 = base64.b64encode(image_bytes).decode('utf-8')
                                chunks.append({
                                    "type": "image",
                                    "content": img_b64,
                                    "page": page_num + 1,
                                    "format": base_image["ext"],
                                    "img_index": img_index
                                })
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to extract image {img_index} from page {page_num + 1}: {str(e)}")
                    except Exception as e:
                        self.logger.warning(f"Failed to process images on page {page_num + 1}: {str(e)}")

            return chunks

        except Exception as e:
            self.logger.error(f"Failed to extract content from PDF {file_path}: {str(e)}")
            raise

    def _safe_extract_table(self, table)->str:
        csc_rows = []
        for row in table.extract():
            processed_row =[]
            for cell in row:
                if cell is None:
                    processed_cell = ""
                elif isinstance(cell,(int,float)):
                    processed_cell =str(cell)
                elif isinstance(cell, str):
                    processed_cell = cell.strip()
                else:
                    processed_cell = str(cell) if cell else ""
                processed_row.append(processed_cell)
            if any(cell for cell in processed_cell):
                csc_rows.append(",".join(processed_row))
        return "\n".join(csc_rows if csc_rows else '')











