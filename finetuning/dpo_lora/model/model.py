import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from trl import DPOConfig, DPOTrainer
from dpo_lora.config.Config import TrainModeArg
from datasets import load_dataset
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_dpo_data(example):
    return {
        "prompt": example['prompt'],
        "chosen": example['chosen'],
        "rejected": example['rejected']
    }

def load_data(confg:TrainModeArg):
    ds = load_dataset(confg.data_name_or_path, split=f'{confg.calibration[:1000]}')
    ds = ds.map(process_dpo_data, remove_columns=ds.column_names)
    return ds


def train(config:TrainModeArg)->tuple:
    ds = load_data(config)
    bnb_config = BitsAndBytesConfig(
        bnb_4bit_quant_type='nf4',
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_use_double_quant=True

    )

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, device_map='auto', trust_remote_code=True, quantization_config=bnb_config)

    tokenizer = AutoTokenizer.from_pretrained(config.args.model_name_or_path,trust_remote_code=True)
    tokenizer.padding_side='right'
    tokenizer.pad_token = tokenizer.eos_token

    dpo_config = DPOConfig(
        output_dir=config.output,
        per_device_train_batch_size=1,
        num_train_epochs=config.num_epoch,
        learning_rate=2e-4,
        logging_steps=5,
        report_to="none",
        max_length=512,
        max_prompt_length=128,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
    )

    peft_config = LoraConfig(
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.1,
        lora_alpha=16
    )

    dpo_trainer = DPOTrainer(
        model,
        args=dpo_config,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=ds,
    )

    logger.info("Starting DPO training...")
    dpo_trainer.train()
    logger.info("Training completed.")




