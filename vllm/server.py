from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, Any, Dict, List
from LLMHelper import LLMServiceManger
import uuid
import logging
import time
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title='OpenAI-Compatible API',
    description='An OpenAI-compatible API for text generation',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 模型定义 ==========
class GenerationRequest(BaseModel):
    """旧版生成请求（保持兼容）"""
    prompt: str = Field(default='hello', description="The input prompt for generation")
    parameters: Optional[Dict[str, Any]] = Field(
        default={
            "temperature": 0.7,
            "max_tokens": 256,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        description="Optional generation parameters"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")


class GenerationResponse(BaseModel):
    """旧版生成响应（保持兼容）"""
    generated_text: str
    request_id: str


class ChatMessage(BaseModel):
    """OpenAI 风格消息"""
    role: str = Field(..., description="The role of the message sender (system, user, assistant)")
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    """OpenAI 风格请求"""
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation so far")
    model: str = Field("default-model", description="The model to use for completion")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="What sampling temperature to use")
    max_tokens: Optional[int] = Field(256, description="The maximum number of tokens to generate")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(0.0, ge=0, le=2, description="Penalty for frequent tokens")
    presence_penalty: Optional[float] = Field(0.0, ge=0, le=2, description="Penalty for new tokens")
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress")


class CompletionChoice(BaseModel):
    """OpenAI 风格选择项"""
    index: int
    message: ChatMessage
    finish_reason: str


class CompletionUsage(BaseModel):
    """OpenAI 风格用量统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI 风格响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ErrorResponse(BaseModel):
    """统一错误响应"""
    error: Dict[str, Any]


# ========== 服务初始化 ==========
service_manager = LLMServiceManger()


# ========== 中间件 ==========
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    request.state.start_time = time.time()

    logger.info(f"Request {request.state.request_id} started: {request.method} {request.url}")

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response
    except Exception as e:
        logger.error(f"Request {request.state.request_id} failed: {str(e)}", exc_info=True)
        raise


# ========== 路由处理 ==========
@app.get("/")
async def root(request: Request):
    return JSONResponse(
        content={
            "message": "Welcome to OpenAI-Compatible API",
            "endpoints": {
                "openai_style": "/v1/chat/completions",
                "legacy": "/generate"
            }
        }
    )


@app.get("/health", tags=["monitoring"])
async def health_check():
    try:
        service = await service_manager.get_service()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": "Service unavailable", "code": 503}}
        )


@app.post("/v1/chat/completions",
          response_model=ChatCompletionResponse,
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse},
              503: {"model": ErrorResponse}
          })
async def create_chat_completion(request: Request, chat_request: ChatCompletionRequest):
    request_id = request.state.request_id
    try:
        service = await service_manager.get_service()
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_request.messages])

        params = {
            "temperature": chat_request.temperature,
            "max_tokens": chat_request.max_tokens,
            "top_p": chat_request.top_p,
            "frequency_penalty": chat_request.frequency_penalty,
            "presence_penalty": chat_request.presence_penalty
        }

        if chat_request.stream:
            stream_generator = await service.generate(prompt=prompt, params=params, stream=True)

            async def generate_stream():
                full_text = ""
                async for text in stream_generator:
                    full_text += text
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": chat_request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                final_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": chat_request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            generated_text = await service.generate(prompt=prompt, params=params, stream=False)
            return JSONResponse(content={
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": chat_request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split())
                }
            })

    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": {"message": str(e), "code": 400}})
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": {"message": "Internal server error", "code": 500}})


@app.post("/generate",
          response_model=GenerationResponse,
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          })
async def generate_text(request: Request, gen_request: GenerationRequest):
    request_id = request.state.request_id
    try:
        service = await service_manager.get_service()

        if gen_request.stream:
            stream_generator = await service.generate(
                prompt=gen_request.prompt,
                params=gen_request.parameters,
                stream=True
            )

            async def generate_stream():
                full_text = ""
                async for text in stream_generator:
                    full_text += text
                    yield text
                logger.info(f"Request {request_id} streamed {len(full_text)} chars")
            return StreamingResponse(generate_stream(), media_type="text/plain")
        else:
            generated_text = await service.generate(
                prompt=gen_request.prompt,
                params=gen_request.parameters,
                stream=False
            )
            return JSONResponse(content={
                "generated_text": generated_text,
                "request_id": request_id
            })

    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": {"message": str(e), "request_id": request_id}
        })
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": {"message": "Internal server error", "request_id": request_id}
        })