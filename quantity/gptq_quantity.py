import logging
import os
import torch
from settings.yaml_loader import ConfigLoader, QuantizationType
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset
import gc
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="auto_gptq")
warnings.filterwarnings("ignore", message="CUDA extension not installed")


class GPTQQuantizer:
    def __init__(self):
        self.config = ConfigLoader().auto_load(config_type=QuantizationType.GPTQ.value, file_path='config.yaml')
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Setup basic logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _prepare_conversation(self, sample):
        """Prepare conversation from dataset sample"""
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')

        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{instruction} {input_text}".strip()},
            {"role": "assistant", "content": output}
        ]

    def load_calibration_data(self, tokenizer, max_length=8192):
        """Load and prepare calibration data for quantization"""
        dataset = load_dataset(
            self.config.calibration_dataset,
            split='train',
            streaming=True
        )

        messages = []
        sample_count = 0
        for sample in dataset:
            if sample_count >= 1000:
                break

            try:
                conversation = self._prepare_conversation(sample)
                messages.append(conversation)
                sample_count += 1
            except Exception as e:
                self.logger.warning(f"Skipping invalid sample: {e}")
                continue

        self.logger.info(f"Loaded {len(messages)} samples for calibration")

        calibration_data = []
        for msg in messages:
            try:
                model_inputs = tokenizer.apply_chat_template(
                    msg,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                input_ids = model_inputs[0, :max_length]
                calibration_data.append({
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids)
                })
            except Exception as e:
                self.logger.warning(f"Skipping sample due to tokenization error: {e}")
                continue

        return calibration_data

    def run_quantization(self):
        """Run the full quantization pipeline"""
        try:
            # Prepare output directory
            os.makedirs(self.config.output_dir, exist_ok=True)

            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True
            )

            # Load calibration data
            self.logger.info("Loading calibration data...")
            calibration_data = self.load_calibration_data(tokenizer)

            # Configure quantization - use direct attribute access instead of .get()
            quantize_config = BaseQuantizeConfig(
                bits=getattr(self.config, 'bits', 4),
                group_size=getattr(self.config, 'group_size', 128),
                model_name_or_path=None,
                model_file_base_name="model",
                desc_act=getattr(self.config, 'desc_act', False),
                static_groups=getattr(self.config, 'desc_act', False),
                sym=getattr(self.config, 'sym', True),
                true_sequential=getattr(self.config, 'true_sequential', True)
            )

            # Load model
            self.logger.info("Loading model for quantization...")
            model = AutoGPTQForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="offload",
                quantize_config=quantize_config
            )

            # Run quantization
            self.logger.info("Starting quantization process...")
            model.quantize(
                calibration_data,
                cache_examples_on_gpu=True
            )

            # Save quantized model
            self.logger.info("Saving quantized model...")
            test_path = "/home/mystic/PycharmProjects/finetuning/ragbi/quantity/qwen3-8b-qptq"
            os.makedirs(test_path, exist_ok=True)
            from pathlib import Path
            tokenizer.save_pretrained(str(self.config.output_dir))
            model.save_quantized(Path(test_path),use_safetensors=True)

            self.logger.info("Quantization completed successfully!")

        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise
        finally:
            # Clean up
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    quantizer = GPTQQuantizer()
    quantizer.run_quantization()