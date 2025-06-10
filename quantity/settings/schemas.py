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
    calibration_dataset: str = Field(
        ...,
        description="Path to calibration dataset (optional)"
    )


QuantizationConfig = AWQConfig  | GPTQConfig 