import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import logging
from dataclasses import dataclass
from typing import Optional, List
from huggingface_hub import login
import os


# 从环境变量读取 token（更安全）
hf_token = os.getenv("HF_TOKEN","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")  # 需提前设置环境变量
if hf_token is not None:
    login(token="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")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_quant_type: str = "nf4"

def load_quantization_config(config: Optional[QuantizationConfig] = None) -> BitsAndBytesConfig:
    """Load quantization configuration with defaults"""
    if config is None:
        config = QuantizationConfig()
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
    )

def load_model_and_tokenizer(
    model_id: str,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    device_map: str = "auto"
) -> tuple:
    """Load model and tokenizer with error handling"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=device_map
        )
        
        # Handle missing pad token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.1
) -> LoraConfig:
    """Create LoRA configuration with sensible defaults"""
    if target_modules is None:
        target_modules = [
            "q_proj", "o_proj", "k_proj", "v_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=lora_dropout,
    )

def load_and_preprocess_data(
    dataset_path: str = "tatsu-lab/alpaca",
    split: str = "train[:20]",
    eos_token: str = "</s>"
):
    """Load and preprocess dataset with error handling"""
    try:
        dataset = load_dataset(dataset_path, split=split)
        
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        def format_text(examples):
            texts = []
            for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
                text = alpaca_prompt.format(instr, inp, out) + eos_token
                texts.append(text)
            return {"text": texts}
        
        return dataset.map(format_text, batched=True)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train(
    model_id: str = "/mnt/c/Users/zhouq/AI学习/Qwen3-8B",
    output_dir: str = "./llama3-lora-finetuned",
    epochs: int = 2,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    max_seq_length: int = 512
):
    """Main training function with comprehensive configuration"""
    try:
        # Set seed for reproducibility
        set_seed(42)
        
        # Step 1: Load data
        logger.info("Loading and preprocessing data...")
        train_dataset = load_and_preprocess_data()
        
        # Step 2: Load quantization config
        logger.info("Loading quantization config...")
        quant_config = load_quantization_config()
        
        # Step 3: Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_id, quant_config)
        
        # Step 4: Load LoRA config
        logger.info("Loading LoRA config...")
        peft_config = load_lora_config()
        
        # Step 5: Configure training arguments
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            num_train_epochs=epochs,
            optim="paged_adamw_8bit",
            warmup_steps=50,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            report_to="none",
            save_total_limit=2,
            evaluation_strategy="no",
            lr_scheduler_type="cosine",
        )
        
        # Step 6: Initialize trainer
        logger.info("Initializing trainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            packing=False,  # Disable packing for better stability
        )
        
        # Step 7: Train and save
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Saving model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training complete! Model saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train()