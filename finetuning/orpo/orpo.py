import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass, field
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
import torch
import gc

@dataclass
class ModelArgs:
    model_name_or_path: str = '/mnt/c/Users/zhouq/AI学习/Qwen2.5-0.5B-Instruct'
    output_dir: str = "output"
    merge_adapter: bool = True


@dataclass
class DataArgs:
    dataset_name_or_path: str = '/mnt/c/Users/zhouq/AI学习/projects/orpo-dpo-mix-40k/data'


@dataclass
class TrainingArgs:
    learning_rate: float = 8e-6
    beta: float = 0.3
    lr_scheduler_type: str = "linear"
    max_length: int = 1024
    optim: str = 'paged_adamw_8bit'
    logging_steps: int = 1
    num_train_epochs: int = 1
    eval_steps: float = 0.2
    warmup_steps: int = 10
    report_to: str = "wandb"
    output_dir: str = "./results/"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4


class ORPODataProcessor:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        batch = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }

        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            batch["prompt"].append(prompt)
            batch["chosen"].append(chosen)
            batch["rejected"].append(rejected)

        # Tokenize all texts
        tokenized_prompt = self.tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        tokenized_chosen = self.tokenizer(
            batch["chosen"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        tokenized_rejected = self.tokenizer(
            batch["rejected"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized_prompt["input_ids"],
            "attention_mask": tokenized_prompt["attention_mask"],
            "chosen_input_ids": tokenized_chosen["input_ids"],
            "chosen_attention_mask": tokenized_chosen["attention_mask"],
            "rejected_input_ids": tokenized_rejected["input_ids"],
            "rejected_attention_mask": tokenized_rejected["attention_mask"],
        }


class ORPOTraining:
    def __init__(self, m_args: ModelArgs, d_args: DataArgs, t_args: TrainingArgs):
        self.m_args = m_args
        self.d_args = d_args
        self.t_args = t_args

    def load_dataset(self):
        ds = load_dataset(self.d_args.dataset_name_or_path, split='train[:200]')

        def format_chat_template(example):
            prompt = example["prompt"] if isinstance(example["prompt"], str) else \
                " ".join([x["content"] for x in example["prompt"] if x["role"] == "user"])

            chosen = self._format_response(example["chosen"])
            rejected = self._format_response(example["rejected"])

            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }

        return ds.map(format_chat_template, batched=False).train_test_split(test_size=0.02)

    def _format_response(self, response):
        if isinstance(response, str):
            return response
        elif isinstance(response, list):
            return " ".join([x["content"] for x in response if x["role"] == "assistant"])
        return str(response)

    def config_models(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            bias="none",
            target_modules=[
                'up_proj', 'down_proj', 'gate_proj',
                'k_proj', 'q_proj', 'v_proj', 'o_proj'
            ],
            task_type="CAUSAL_LM"
        )

        orpo_config = ORPOConfig(
            learning_rate=self.t_args.learning_rate,
            beta=self.t_args.beta,
            lr_scheduler_type=self.t_args.lr_scheduler_type,
            max_length=self.t_args.max_length,
            optim=self.t_args.optim,
            logging_steps=self.t_args.logging_steps,
            num_train_epochs=self.t_args.num_train_epochs,
            eval_steps=self.t_args.eval_steps,
            warmup_steps=self.t_args.warmup_steps,
            report_to=self.t_args.report_to,
            output_dir=self.t_args.output_dir,
            per_device_train_batch_size=self.t_args.per_device_train_batch_size,
            per_device_eval_batch_size=self.t_args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.t_args.gradient_accumulation_steps,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.m_args.model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

        model = prepare_model_for_kbit_training(model)
        tokenizer = AutoTokenizer.from_pretrained(self.m_args.model_name_or_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer, orpo_config, peft_config

    def train_models(self):
        model, tokenizer, orpo_config, peft_config = self.config_models()
        ds = self.load_dataset()

        # Create the data processor
        processor = ORPODataProcessor(tokenizer, orpo_config.max_length)

        trainer = ORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            peft_config=peft_config,
            data_processor=processor,  # Pass the processor here
        )

        trainer.train()
        self._save_model(model, tokenizer)
        trainer.save_model(os.path.join(self.m_args.output_dir, "final_model"))
        del trainer
        self.cleanup()

    def _save_model(self, model, tokenizer):
        os.makedirs(self.m_args.output_dir, exist_ok=True)
        model.save_pretrained(self.m_args.output_dir)
        tokenizer.save_pretrained(self.m_args.output_dir)

        if self.m_args.merge_adapter:
            try:
                merged_model = model.merge_and_unload()
                merged_dir = os.path.join(self.m_args.output_dir, "merged")
                os.makedirs(merged_dir, exist_ok=True)
                merged_model.save_pretrained(merged_dir)
                tokenizer.save_pretrained(merged_dir)
            except Exception as e:
                print(f"Failed to merge adapter: {e}")

    def cleanup(self):
        torch.cuda.empty_cache()

    def run(self):
        try:
            self.train_models()
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        finally:
            self.cleanup()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()
    instance = ORPOTraining(ModelArgs(), DataArgs(), TrainingArgs())
    instance.run()