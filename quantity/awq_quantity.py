import gc
import logging
from pathlib import Path
import torch.cuda
from awq import AutoAWQForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer
from settings.yaml_loader import ConfigLoader
from tqdm import tqdm
import torch
from contextlib import contextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMQuantizer:
    """AWQ模型量化处理器（生产级实现）"""

    def __init__(self, config_path: str = "config.yaml"):
        self.quant_config = ConfigLoader().auto_load(config_type='awq', file_path=config_path)
        logger.info(f"Loaded config: {self.quant_config}")
        self._validate_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.__init_cuda()

    def __init_cuda(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    def _validate_config(self) -> None:
        """验证配置合法性"""
        required_attrs = ['model_name_or_path', 'output_dir', 'w_bit']
        for attr in required_attrs:
            if not hasattr(self.quant_config, attr):
                raise ValueError(f"Missing required config attribute: {attr}")

        if not (2 <= self.quant_config.w_bit <= 8):
            raise ValueError("w_bit must be between 2 and 8")

    @contextmanager
    def _gpu_memory_context(self):
        try:
            yield
        finally:
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()

    def load_calibration_data(self):
        """加载并预处理校准数据"""
        try:
            logger.info("Loading calibration data...")

            # 加载数据集
            dataset = load_dataset(
                self.quant_config.calibration_dataset,
                split='train',
                streaming=True
            )

            # 收集文本样本
            texts = []
            progress = tqdm(total=32, desc="Collecting calibration samples")

            for sample in dataset:
                if len(texts) >= 32:
                    break

                # 处理不同格式的数据
                if isinstance(sample, dict):
                    # 处理字典格式的数据
                    if 'text' in sample:
                        texts.append(sample['text'])
                    elif 'content' in sample:
                        texts.append(sample['content'])
                    elif 'conversations' in sample:
                        # 处理对话格式
                        conv = sample['conversations']
                        text = "\n".join([turn['value'] for turn in conv if isinstance(turn, dict) and 'value' in turn])
                        if text:
                            texts.append(text)
                elif isinstance(sample, str):
                    # 直接处理字符串
                    texts.append(sample)

                progress.update(1)

            progress.close()

            if not texts:
                raise ValueError("No valid text samples found in calibration data")

            logger.info(f"Collected {len(texts)} calibration samples")
            return texts

        except Exception as e:
            logger.error(f"Error loading calibration data: {str(e)}")
            raise

    def quantize_model(self) -> None:
        """执行完整的量化流程"""
        try:
            # 1. 创建输出目录
            output_dir = Path(self.quant_config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 2. 加载tokenizer和模型
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.quant_config.model_name_or_path,
                trust_remote_code=True
            )

            logger.info("Loading model...")
            model = AutoAWQForCausalLM.from_pretrained(
                self.quant_config.model_name_or_path,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                offload_folder="offload",
            )

            # 3. 准备量化配置
            quant_config = {
                "zero_point": getattr(self.quant_config, "zero_point", True),
                "q_group_size": getattr(self.quant_config, "q_group_size", 128),
                "w_bit": self.quant_config.w_bit,
            }
            logger.info(f"Quantization config: {quant_config}")

            # 4. 加载校准数据
            calibration_data = self.load_calibration_data()

            # 确保数据不为空
            if not calibration_data:
                raise ValueError("No calibration data available for quantization")

            # 5. 执行量化
            logger.info("Starting quantization process...")
            model.quantize(
                tokenizer,
                quant_config=quant_config,
                calib_data=calibration_data,
                n_parallel_calib_samples=2,
                max_calib_samples=16,
                max_calib_seq_len=256  # 适当增加序列长度
            )

            # 6. 保存模型
            logger.info("Saving quantized model...")
            model.save_quantized(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            logger.info(f"Model successfully saved to {output_dir}")

        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}", exc_info=True)
            raise
        finally:
            # 确保资源释放
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    try:
        quantizer = LLMQuantizer()
        quantizer.quantize_model()
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise