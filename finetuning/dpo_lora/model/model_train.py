import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,PeftModel
)
from trl import DPOConfig, DPOTrainer

from datasets import load_dataset
import logging
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DPOTrainerPipeline:
    def __init__(self,config:TrainModeArg):
        self.config =  config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def _setup_quantization(self):
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_quant_type='nf4',
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, 'bfloat16'),
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_use_double_quant=True

        )
        return bnb_config

    def _load_data(self,tokenizer):
        """修正后的数据加载方法"""
        ds = load_dataset(self.config.data_name_or_path,
                          split=f'train[:1000]' )

        def format_dpo_sample(example):
            # 假设数据结构是对话式：[{"content":..., "role":...}, ...]
            if isinstance(example["prompt"], list):  # 如果已经是列表格式
                prompt = " ".join([x["content"] for x in example["prompt"] if x["role"] == "user"])
                chosen = " ".join([x["content"] for x in example["chosen"] if x["role"] == "assistant"])
                rejected = " ".join([x["content"] for x in example["rejected"] if x["role"] == "assistant"])
            else:  # 如果是单条数据
                prompt = example["prompt"]
                chosen = example["chosen"]
                rejected = example["rejected"]

            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }

        def tokenize_function(examples):
            # Tokenize prompts
            tokenized_prompts = tokenizer(
                examples["prompt"],
                truncation=True,
                max_length=self.config.max_prompt_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Tokenize responses（考虑预留prompt空间）
            max_response_length = self.config.max_prompt_length
            tokenized_chosen = tokenizer(
                examples["chosen"],
                truncation=True,
                max_length=max_response_length,
                padding="max_length",
                return_tensors="pt"
            )
            tokenized_rejected = tokenizer(
                examples["rejected"],
                truncation=True,
                max_length=max_response_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids_prompt": tokenized_prompts["input_ids"],
                "attention_mask_prompt": tokenized_prompts["attention_mask"],
                "input_ids_chosen": tokenized_chosen["input_ids"],
                "attention_mask_chosen": tokenized_chosen["attention_mask"],
                "input_ids_rejected": tokenized_rejected["input_ids"],
                "attention_mask_rejected": tokenized_rejected["attention_mask"],
            }

        ds= ds.map(format_dpo_sample, remove_columns=ds.column_names)
        # ds = ds.map(tokenize_function)
        return ds

    def _initialize_components(self):
        """初始化模型、tokenizer和训练器"""
        # 1. 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=self._setup_quantization(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=getattr(torch, "bfloat16"),
            attn_implementation="flash_attention_2"
        )
        model = prepare_model_for_kbit_training(model)

        # 2. 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 3. 配置LoRA
        peft_config = LoraConfig(
            r=self.config.lora_rank or 32,
            lora_alpha=self.config.lora_alpha or 16,
            target_modules=["q_proj", "v_proj"],  # 针对LLaMA结构
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 4. DPO配置
        dpo_config = DPOConfig(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size or 2,
            gradient_accumulation_steps=self.config.grad_accum_steps or 4,
            num_train_epochs=self.config.num_epochs or 3,
            learning_rate=self.config.lr or 5e-5,
            logging_steps=10,
            save_steps=500,
            optim="paged_adamw_8bit",
            max_length=1024,
            max_prompt_length=512,
            remove_unused_columns=False,
            report_to="tensorboard",
            beta=0.3,  # 当前可能>0.5，尝试调低
            loss_type="sigmoid"  # 平滑奖励差异,
        )

        return model, tokenizer, peft_config, dpo_config

    def train(self):
        """完整的DPO训练流程"""
        # 1. 初始化组件
        model, tokenizer, peft_config, dpo_config = self._initialize_components()
        train_dataset = self._load_data(tokenizer)

        # 2. 初始化DPO Trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

        # 3. 训练
        logger.info("Starting DPO training...")
        trainer.train()

        # 4. 保存模型
        self._save_model(trainer.model, tokenizer)
        logger.info(f"Model saved to {self.config.output_dir}")

        return trainer

    def _save_model(self, model, tokenizer):

        os.makedirs(self.config.output_dir, exist_ok=True)

        # 保存适配器
        model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        # 合并LoRA权重（可选生产部署）
        if self.config.merge_adapter:
            merged_model = model.merge_and_unload()
            merged_dir = os.path.join(self.config.output_dir, "merged")
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)

    @classmethod
    def load_for_inference(cls, model_path, merge_adapter=True):
        """加载训练好的模型进行推理"""
        # 1. 加载基础组件
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 2. 加载模型
        if merge_adapter and os.path.exists(os.path.join(model_path, "merged")):
            # 加载合并后的模型
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(model_path, "merged"),
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        else:
            # 加载原始模型+适配器
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            if merge_adapter:
                model = model.merge_and_unload()

        return model, tokenizer


# 使用示例
if __name__ == "__main__":
    # 1. 训练阶段
    config = TrainModeArg()

    trainer = DPOTrainerPipeline(config)
    trainer.train()

    # 2. 推理测试
    model, tokenizer = DPOTrainerPipeline.load_for_inference("./output")

    test_prompt = "Instruction: Write an engaging YouTube title\nInput: habituation in causal inference"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id  # 重要：设置pad token
    )

    # 4. 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

    print("Generated:", generated_text[0])