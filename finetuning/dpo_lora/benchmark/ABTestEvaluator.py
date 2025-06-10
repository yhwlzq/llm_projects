import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

class ABTestEvaluator:
    def __init__(self, base_model_path, dpo_model_path):
        self.base_model, self.base_tokenizer = self._load_model(base_model_path)
        self.dpo_model, self.dpo_tokenizer = self._load_model()

    def _load_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained( model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16)

        return model, tokenizer

    def generate_response(self, prompt, **gen_kwargs):
        base_inputs = self.base_tokenizer(prompt, return_tensors='pt').to(self.base_model.device)
        dpo_inputs = self.dpo_tokenizer(prompt, return_tensors="pt").to(self.dpo_model.device)

        base_output = self.base_model.generate(**base_inputs, **gen_kwargs)
        dpo_output = self.dpo_model.generate(**dpo_inputs, **gen_kwargs)

        return {
            "base": self.base_tokenizer.decode(base_output[0], skip_special_tokens=True),
            "dpo": self.dpo_tokenizer.decode(dpo_output[0], skip_special_tokens=True)
        }

    def automatic_evaluation(self, test_dataset, num_samples=50, **gen_kwargs):
        from evaluate import load
        rouge = load('rouge')
        bleurt = load("bleurt", "bleurt-large-512")
        result = []




