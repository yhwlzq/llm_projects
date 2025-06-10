import threading
import os
import yaml
from typing import TypeVar, Type
from pathlib import Path
from settings.schemas import GPTQConfig, AWQConfig

from enum import Enum

T = TypeVar("T", bound='QuantizationConfig')

class QuantizationType(str, Enum):
    AWQ = "awq"
    GPTQ = "gptq"


class ConfigLoader(object):

    _config_registry = {
        QuantizationType.AWQ: AWQConfig,
        QuantizationType.GPTQ: GPTQConfig,
    }

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _load_yaml(file_path:str|Path)->dict:
        file_path = os.path.join(Path(__file__).parent,file_path)
        with open(file_path,'r',encoding='utf-8')as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"yaml parser error")

    def load_config(self,file_path:Path, config_type:Type[T]):
        raw_data = self._load_yaml(file_path=file_path)
        try:
            return raw_data[config_type]
        except Exception as e:
            raise ValueError(f"配置验证失败: {str(e)}")

    def auto_load(self,  config_type:QuantizationType=QuantizationType.AWQ,file_path:Path="config.yaml"):
        class_name = self._config_registry.get(config_type)
        if not class_name:
            raise ValueError(f"未注册的配置类型: {config_type}")
        try:
            return class_name(**self.load_config(file_path, config_type))
        except Exception as e:
            raise ValueError(f"配置验证失败: {str(e)}")
