from typing import Dict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import re

class LLM_Chunker:
    def __init__(self,config:Dict):
        self.chunk_size = config.get("chunk_size", 512)
        self.overlap = config.get("overlap", 50)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer", "bert-base-uncased"))
        self.embedding_model = SentenceTransformer(config.get("embedding_model", "all-MiniLM-L6-v2"))
        self.llm_api = config.get("llm_api")
        self.min_chunk_size = config.get("min_chunk_size", 64)


    
    def _remove_invisible_chars(self, text: str) -> str:
        """移除不可见字符和零宽字符"""
        # Unicode中的不可见控制字符
        invisible_chars = re.compile(r'[\u200b-\u200f\u202a-\u202e\ufeff]')
        return invisible_chars.sub('', text)

    def _normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        # 替换所有空白字符（包括全角空格）为普通空格
        text = re.sub(r'[\s\u3000]+', ' ', text)
        return text.strip()
        
        
    def _remove_control_chars(self, text: str) -> str:
        """移除控制字符"""
        return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    def _normalize_chinese_punctuation(self, text: str) -> str:
        """中文标点规范化"""
        # 重复标点处理（！！！→！）
        text = re.sub(r'([！？。])\1+', r'\1', text)
        # 英文标点转中文标点
        punctuation_map = {
            ',': '，',
            ':': '：',
            ';': '；',
            '?': '？',
            '!': '！'
        }
        for eng, chn in punctuation_map.items():
            text = text.replace(eng, chn)
        return text
    
    
    def _handle_chinese_english_mix(self, text: str) -> str:
        """处理中英文混排"""
        # 中英文之间加空格（但保留已有空格情况）
        text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([\u4e00-\u9fa5])', r'\1 \2', text)
        # 处理数字与中文混排
        text = re.sub(r'([\u4e00-\u9fa5])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([\u4e00-\u9fa5])', r'\1 \2', text)
        return text

    def _remove_special_chinese_chars(self, text: str) -> str:
        """移除特殊中文字符"""
        # 非常用汉字、特殊符号等
        special_chars = re.compile(r'[〇㐀䶵𠀀-𪛔𪜀-𫜴𫝀-𫠝𫠠-𬺡𬺰-𮯠]')
        return special_chars.sub('', text)

    def _normalize_unicode(self, text: str) -> str:
        """Unicode规范化"""
        import unicodedata
        # 标准化为NFKC格式（兼容组合字符）
        text = unicodedata.normalize('NFKC', text)
        # 处理特殊空格
        text = text.replace('\u3000', ' ')  # 全角空格
        text = text.replace('\u00a0', ' ')  # 不换行空格
        return text
    

    
    def _convert_fullwidth_chars(self, text: str) -> str:
        """全角字符转半角"""
        result = []
        for char in text:
            code = ord(char)
            # 全角字母数字转半角
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            # 全角空格转半角
            elif code == 0x3000:
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)


    def _process_text(self, text:str)->str:
        """增强版中文文本预处理
        功能：
        1. 基础清洗
        2. 特殊字符处理
        3. 中文特定处理
        4. 编码规范化
        
        参数：
            text: 原始输入文本
            
        返回：
            预处理后的干净文本
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # 1. 基础清洗
        text = self._remove_invisible_chars(text)
        text = self._normalize_whitespace(text)
        
        # 2. 特殊内容处理
        text = self._remove_urls_emails(text)
        text = self._remove_control_chars(text)
        
        # 3. 中文特定处理
        text = self._normalize_chinese_punctuation(text)
        text = self._handle_chinese_english_mix(text)
        text = self._remove_special_chinese_chars(text)

            # 4. 编码规范化
        text = self._normalize_unicode(text)
        text = self._convert_fullwidth_chars(text)
        
        return text.strip()

