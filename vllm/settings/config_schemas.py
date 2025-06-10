from pydantic import BaseModel, Field
from typing import Literal

class BaseQuantizationConfig(BaseModel):
    """Base configuration for model quantization"""
    model_name_or_path: str = Field(
        ...,
        description="Path to the original model directory or HuggingFace model ID",
        examples=["/path/to/model", "Qwen/Qwen1.5-7B"]
    )
    output_dir: str = Field(
        default="./quantized_models",
        description="Directory to save quantized model"
    )

class AWQConfig(BaseQuantizationConfig):
    """Configuration for Activation-aware Weight Quantization"""
    zero_point: bool = Field(
        default=False,
        description="Enable zero-point quantization for better accuracy at low bits"
    )
    w_bit: Literal[2, 3, 4, 8] = Field(
        default=4,
        description="Quantization bit width (2/3/4/8)"
    )
    q_group_size: int = Field(
        default=64,
        ge=64, le=256,
        description="Quantization group size (smaller = more accurate but slower)"
    )
    quant_type: Literal["awq"] = Field(
        "awq",
        description="Quantization method type"
    )
    calibration_dataset: str = Field(
        ...,
        description="Path to calibration dataset (optional)"
    )

class GPTQConfig(BaseQuantizationConfig):
    """Configuration for GPTQ Quantization"""
    damp_percent: float = Field(
        default=0.1,
        ge=0.01, le=1.0,
        description="Damping percentage for numerical stability"
    )
    blocksize: int = Field(
        default=128,
        description="Block size for quantization"
    )
    quant_type: Literal["gptq"] = Field(
        "gptq",
        description="Quantization method type"
    )


class vLLMEngineArgs(BaseQuantizationConfig):

    tensor_parallel_size: int = Field(
        default=1,
        description="tensor_parallel_size"
    )

    quantization:str = Field(
        default='awq', description="quantization"
    )
    max_num_seqs:int  = Field(
        default=64, description="max num seqs"
    )
    max_model_len:int = Field(
        default=1024, description="max model len"
    )
    tensor_parallel_size:int = Field(
        default=1, description="tensor_parallel_size"
    )
    trust_remote_code:bool =  Field(
        default=True, description="trust_remote_code"
    )
    gpu_memory_utilization:float = Field(
        default=0.85, description="trust_remote_code"
    )


class vLLM_Sampling(BaseQuantizationConfig):

    early_stopping: int = Field(
        default=False,
        description="early_stopping"
    )

    top_p:float = Field(
        default=0.9, description="top_p"
    )
    temperature:int  = Field(
        default=64, description="max num seqs"
    )
    max_model_len:int = Field(
        default=1024, description="max model len"
    )
    tensor_parallel_size:int = Field(
        default=1, description="tensor_parallel_size"
    )
    trust_remote_code:bool =  Field(
        default=True, description="trust_remote_code"
    )
    gpu_memory_utilization:float = Field(
        default=0.85, description="trust_remote_code"
    )

QuantizationConfig = AWQConfig  | GPTQConfig |vLLMEngineArgs 