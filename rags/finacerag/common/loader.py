import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datasets import Dataset, DatasetDict, Value, load_dataset

logger = logging.getLogger(__name__)

class HFDataLoader:
    def __init__(
        self,
        hf_repo: Optional[str] = None,
        data_folder: Optional[str] = None,
        subset: Optional[str] = None,
        prefix: Optional[str] = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        keep_in_memory: bool = False,
        streaming: bool = False
    ):
        """
        初始化数据加载器
        
        Args:
            hf_repo: Hugging Face 仓库名称
            data_folder: 本地数据文件夹路径
            subset: 数据集子集名称
            prefix: 文件前缀
            corpus_file: 语料库文件名
            query_file: 查询文件名
            keep_in_memory: 是否将数据保留在内存中
            streaming: 是否使用流式加载
        """
        self.corpus: Optional[Dataset] = None
        self.queries: Optional[Dataset] = None
        self.hf_repo = hf_repo
        self.subset = subset
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory

        if hf_repo:
            logger.warning("A Hugging Face repository is provided. This will override the data_folder, prefix, and *_file arguments.")
        else:
            if data_folder is None or subset is None:
                raise ValueError("A Hugging Face repository or local directory is required.")
            
            if prefix:
                query_file = f"{prefix}_{query_file}"
            
            self.corpus_file = self._build_file_path(data_folder, subset, corpus_file)
            self.query_file = self._build_file_path(data_folder, subset, query_file)

    @staticmethod
    def _build_file_path(data_folder: Optional[str], subset: str, filename: str) -> str:
        """构建完整的文件路径"""
        if data_folder:
            return (Path(data_folder) / subset / filename).as_posix()
        return filename

    @staticmethod
    def _validate_file(file_path: str, expected_ext: str = "jsonl"):
        """验证文件是否存在且扩展名正确"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File {file_path} not found! Please provide a valid file.")
        if not file_path.endswith(expected_ext):
            raise ValueError(f"File {file_path} must have the extension {expected_ext}")

    def _load_from_hf(self, split: str) -> Dataset:
        """从Hugging Face加载数据集"""
        logger.info("Loading %s from Hugging Face...", split)
        ds = load_dataset(
            path=self.hf_repo,
            name=self.subset,
            split=split,
            keep_in_memory=self.keep_in_memory,
            streaming=self.streaming
        )
        return self._process_dataset(ds, split)

    def _load_from_local(self, file_path: str, split: str) -> Dataset:
        """从本地文件加载数据集"""
        self._validate_file(file_path)
        logger.info("Loading %s from local file...", split)
        ds = load_dataset(
            "json",
            data_files=file_path,
            streaming=self.streaming,
            keep_in_memory=self.keep_in_memory
        )
        # 如果是非流式加载，需要从DatasetDict中获取数据集
        if not self.streaming and isinstance(ds, DatasetDict):
            ds = ds["train"]  # 默认使用"train" split
        return self._process_dataset(ds, split)

    def _process_dataset(self, dataset: Dataset, split: str) -> Dataset:
        """处理数据集，统一格式"""
        required_columns = {"id", "text"}
        if split == "corpus":
            required_columns.add("title")
        
        # 确保_id字段存在并转换为string类型
        if "_id" in dataset.column_names:
            dataset = dataset.rename_column("_id", "id")
        
        if "id" not in dataset.column_names:
            raise ValueError(f"Dataset {split} must contain an 'id' column")
        
        dataset = dataset.cast_column("id", Value("string"))
        
        # 移除不需要的列
        columns_to_keep = [col for col in dataset.column_names if col in required_columns]
        columns_to_remove = [col for col in dataset.column_names if col not in required_columns]
        
        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)
        
        return dataset

    def load_corpus(self) -> Dataset:
        """加载语料库数据集"""
        if self.corpus is None:
            if self.hf_repo:
                self.corpus = self._load_from_hf("corpus")
            else:
                self.corpus = self._load_from_local(self.corpus_file, "corpus")
            logger.info("Loaded %d documents from corpus.", len(self.corpus))
        return self.corpus

    def load_queries(self) -> Dataset:
        """加载查询数据集"""
        if self.queries is None:
            if self.hf_repo:
                self.queries = self._load_from_hf("queries")
            else:
                self.queries = self._load_from_local(self.query_file, "queries")
            logger.info("Loaded %d queries.", len(self.queries))
        return self.queries

    def load(self) -> Tuple[Dataset, Dataset]:
        """加载语料库和查询数据集"""
        corpus = self.load_corpus()
        queries = self.load_queries()
        
        logger.info("Corpus example: %s", corpus[0] if not self.streaming else next(iter(corpus)))
        logger.info("Query example: %s", queries[0] if not self.streaming else next(iter(queries)))
        
        return corpus, queries