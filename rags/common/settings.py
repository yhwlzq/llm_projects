import os
from enum import Enum
from pathlib import Path

class FileType(Enum):
    PDF = 'pdf'
    CSV = 'csv'
    #IMAGE ='img'


class Settings:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_DIR = BASE_DIR / "chroma_db"
    CACHE_DIR = BASE_DIR / "cache"

    EMBEDDING_MODEL = "/mnt/c/Users/zhouq/AI学习/all-MiniLM-L6-v2"
    COLLECTION_PREFIX = "rag"

    TEXT_EMBEDDING_MODEL = 'text-embedding-3-small'
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TEXT_CHUNK_SIZE = 500
    TEXT_CHUNK_OVERLAB = 100
    TABLE_CHUNK_SIZE = 500
    EXTRACT_IMAGE =  True

    CLIP_MODEL = "/mnt/c/Users/zhouq/AI学习/projects/clip-vit-base-patch32"

    for dir in [DATA_DIR, CHROMA_DIR, CACHE_DIR]:
        os.makedirs(dir, exist_ok=True)


settings = Settings()
