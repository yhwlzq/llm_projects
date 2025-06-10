from datasets import load_dataset
from transformers import AutoTokenizer

from dpo_lora.config.Config import  TrainModeArg


class DataProcessor:
    instance=None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls,*args, **kwargs)
            cls.instance.init_tokenizer()
        return cls.instance

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TrainModeArg.model_name_or_path)

    def format_prompt(self,example):
        return {
            "prompt":  [ ex['content'] for ex in  example["prompt"] ],
            "chosen": [ ex['content'] for ex in  example["chosen"] ],
            "rejected": [ ex['content'] for ex in  example["rejected"] ],
        }


    def process_dpo_data(self, example):
        return {
            "prompt":example['prompt'],
            "chosen":example['chosen'],
            "rejected":example['rejected']
        }

    def load_data(self) :
        ds = load_dataset(TrainModeArg.data_name_or_path,split=f'{TrainModeArg.calibration[:1000]}')
        ds = ds.map(self.format_prompt, remove_columns=ds.column_names)
        return ds


b = DataProcessor().load_data()
print(b[:5])
