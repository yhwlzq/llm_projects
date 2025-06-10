from dataclasses import dataclass

@dataclass
class TrainModeArg:
    model_name_or_path = "/mnt/c/Users/zhouq/AI学习/projects/Qwen2.5-0.5B-Instruct"

    data_name_or_path:str = "/mnt/c/Users/zhouq/AI学习/projects/youtube-titles-dpo/data"
    calibration:str ='train'

    output_dir:str = "output"
    num_epochs:int=5
    max_prompt_length:int=1024

    merge_adapter:bool = True
    lora_rank:int=32
    lora_alpha:int=16
    batch_size:int=8
    grad_accum_steps:int =4
    lr:float = 2e-5
