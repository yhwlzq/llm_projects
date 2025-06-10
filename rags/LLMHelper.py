import os
from vllm.engine.arg_utils import  AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from settings.yaml_loader import ConfigLoader, QuantizationType
from settings.config_schemas import vLLMEngineArgs
from typing import Optional
import asyncio
import logging
import uuid
from threading import Lock

os.environ['VLLM_USE_MODELSCOPE'] = 'False'
os.environ['HF_HUB_OFFLINE'] = '1'

class LLMService(object):
    def __init__(self):
        self.logger = logging.getLogger("llm_service")
        self.logger.setLevel(logging.INFO)
        self.config:vLLMEngineArgs = ConfigLoader().auto_load(QuantizationType.MODEL_ARGS)
        self.default_sampling_params = self._init_default_sampling()

    def _init_default_engine(self):
        self.engine_args = AsyncEngineArgs(
            model= self.config.model_name_or_path,
            tokenizer=self.config.model_name_or_path,  # Same path if tokenizer is in the same directory
            trust_remote_code=True,  # Required for Qwen models
            max_num_seqs=64,
            max_model_len=1024,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            dtype='float16',
            enforce_eager=True
        )
        self.engine = None

    def _init_default_sampling(self) ->SamplingParams:
        return SamplingParams(
            temperature=0.3,
            top_p=0.9,
            top_k=1,
            repetition_penalty=1.0,
            max_tokens=1024,

        )

    def initialize(self):
        self._init_default_engine()
        try:
            if self.engine is not None:
                self.logger.warning("Engine already started")
                return
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            self.logger.info('Engine started successfully')
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                self.logger.warning("Retrying with reduced memory utilization...")
                self.logger.error(f"Engine initialization failed: {str(e)}")
                raise

    def _warmup_model(self) ->None:
        warm_text = "Model warmup"
        samplings_params = SamplingParams(max_tokens=10)
        try:
            self.engine.generate(warm_text, samplings_params)
            self.logger.info("Warmup completed")
        except Exception as e:
            self.logger.warning(f"warmup failed :{str(e)}")

    async def generate(
            self,
            prompt: str,
            params: Optional[dict] = None,
            stream: bool = False
    ) -> str:
        """
        优化后的生成方法，支持流式和一次性输出

        Args:
            prompt: 输入的提示文本
            params: 生成参数字典
            stream: 是否使用流式输出

        Returns:
            流式模式下返回异步生成器
            非流式模式下返回最终生成的文本

        Raises:
            RuntimeError: 引擎未初始化或生成失败
            ValueError: 输入参数无效
        """
        # 参数验证
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        
        print(params)

        # 解析生成参数
        sampling_params = self._resolve_sampling_params(params)
        print(sampling_params)
        request_id = f"req_{uuid.uuid4().hex}"  # 使用UUID替代hash更可靠

        try:
            # 获取生成器
            results_generator = self.engine.generate(
                prompt,
                sampling_params,
                request_id
            )

            # 流式输出处理
            if stream:
                async def stream_generator():
                    try:
                        async for output in results_generator:
                            if output.outputs:
                                yield output.outputs[0].text
                    except Exception as e:
                        self.logger.error(f"Stream generation failed: {str(e)}")
                        raise

                return stream_generator()

            # 非流式输出处理
            else:
                final_output = []
                async for output in results_generator:
                    if output.outputs:
                        text = output.outputs[0].text
                        final_output.append(text)
                        self.logger.debug(f"Intermediate output: {text}")

                if not final_output:
                    raise RuntimeError("No output generated")

                # 合并所有输出片段
                return "".join(final_output)

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Generation failed: {str(e)}") from e

    def _resolve_sampling_params(self, params: Optional[dict]):
        defaults = {
            'temperature': 0.3,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.5,
            'frequency_penalty': 0.5,
            'presence_penalty': 0.3,
            'max_tokens': 512
        }
        if params:
            defaults.update({k: v for k, v in params.items() if k in defaults})
        return SamplingParams(**defaults)

    # def _resolve_sampling_params(self, params:Optional[dict]=None):
    #     if params is None:
    #         return self.default_sampling_params
    #     else:
    #         final_params = SamplingParams(
    #             temperature=self.default_sampling_params.temperature,
    #             top_p=self.default_sampling_params.top_p,
    #             top_k=self.default_sampling_params.top_k,
    #             repetition_penalty=self.default_sampling_params.repetition_penalty,
    #             max_tokens=self.default_sampling_params.max_tokens,
    #         )
    #         for key, value in params.items():
    #             if hasattr(final_params, key):
    #                 setattr(final_params, key, value)
    #             else:
    #                 self.logger.warning(f"Ignoring invalid sampling param: {key}")
    #         return final_params

    async def shutdown(self) -> None:
        """关闭引擎"""
        if self.engine is not None:
            del self.engine
            self.engine = None
            self.logger.info("Engine shutdown successfully")

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

class LLMServiceManger:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.service = None
        return cls._instance


    async def get_service(self) ->LLMService:
        if self.service is None:
            self.service = LLMService()
            self.service.initialize()
        return self.service

async def main():
    try:
        service_manager = LLMServiceManger()
        service = await service_manager.get_service()
        response1 = await service.generate(
            "写一个Python贪吃蛇游戏",
            {"temperature": 0.7, "max_tokens": 512,"repetition_penalty":1.2}
        )
        print("Generated:", response1)

        response2 = await service.generate(
            "写一个Python贪吃蛇游戏",
            {"temperature": 0.3, "max_tokens": 512,"repetition_penalty":1.2}
        )
        print("Generated:", response2)
    except Exception as e:
        logging.error(f"Service error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())








