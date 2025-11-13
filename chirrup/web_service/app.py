import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from pydantic import BaseModel, Field

from chirrup.web_service.config import get_config
from chirrup.engine_core import AsyncEngineCore
from chirrup.core_structure import ModelLoadConfig, DEFAULT_STOP_TOKENS
from chirrup.interface import AsyncEngineCompletion
from chirrup.utils.prompt_formatters import (
    format_openai_message_quick_thinking,
    format_openai_message_no_thinking,
    format_openai_message_with_thinking,
)
from chirrup.utils.streaming_string_parser import (
    StreamingStringParser,
    TRIE_THINK_NO_TRIGGER,
)
from chirrup.utils.state_cache import SimpleStateCache
from chirrup.web_service.api_model import (
    ModelsResponse,
    ModelInfo,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatCompletionResponseChoice,
    ChatCompletionResponseUsage,
    ChatMessage,
    ErrorResponse,
    # Alic 翻译接口
    TranslateRequest,
    TranslateResponse,
    TranslationResult,
    # Rollout 接口
    RolloutRequest,
    RolloutStreamChoice,
    RolloutStreamResponse,
)


# 全局引擎实例
engine_core: Optional[AsyncEngineCore] = None
state_cache: SimpleStateCache = None

model_list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_core, model_list, state_cache

    print("正在初始化 AsyncEngineCore...")

    try:
        # 加载配置
        config = get_config()
        print(f"使用配置: 模型路径={config.model_path}, Worker数量={config.worker_num}, 批处理大小={config.batch_size}")

        # 创建引擎核心
        engine_core = AsyncEngineCore()

        # 创建模型配置
        model_config = ModelLoadConfig(
            model_path=config.model_path,
            vocab_path=config.vocab_path,
            vocab_size=config.vocab_size,
            head_size=config.head_size,
        )
        state_cache = SimpleStateCache(config.state_cache_size)

        # 初始化 Worker
        print("正在加载模型...")
        engine_core.init(worker_num=config.worker_num, model_config=model_config, batch_size=config.batch_size)

        model_list = [
            ModelInfo(id="rwkv-latest", created=int(time.time()), owned_by="chirrup"),
            ModelInfo(id="rwkv-latest:thinking", created=int(time.time()), owned_by="chirrup"),
            ModelInfo(id="rwkv-latest:no-thinking", created=int(time.time()), owned_by="chirrup"),
        ]

        # 将引擎实例存储到应用状态中
        app.state.engine_core = engine_core
        app.state.config = config

        yield

    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise
    finally:
        # 清理资源
        print("正在清理资源...")
        if engine_core:
            engine_core.shutdown()
            print("✓ 资源清理完成")


# 创建 FastAPI 应用
app = FastAPI(
    title="RWKV OpenAI Compatible API",
    description="RWKV 大模型推理服务，兼容 OpenAI API 格式",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "model_loaded": engine_core is not None,
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """列出可用模型"""
    return ModelsResponse(data=model_list)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, connection: Request):
    """创建聊天完成，支持流式和非流式响应"""

    if not engine_core:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        model_name_list = request.model.split(":")
        if "thinking" in model_name_list:
            cache_prefill_padding = 2
            prompt = format_openai_message_with_thinking(request.messages)
        elif "no-thinking" in model_name_list:
            prompt = format_openai_message_no_thinking(request.messages)
            cache_prefill_padding = 0
        else:
            prompt = format_openai_message_quick_thinking(request.messages)
            cache_prefill_padding = 6

        # print(f"'''{prompt}'''")
        # 编码输入
        prefill_tokens = [0] if request.pad_zero else []
        prefill_tokens += engine_core.tokenizer.encode(prompt)

        # 处理停止词
        stop_tokens = []
        if request.stop:
            if isinstance(request.stop, str):
                stop_tokens = engine_core.tokenizer.encode(request.stop)
            else:
                for stop_word in request.stop:
                    stop_tokens.extend(engine_core.tokenizer.encode(stop_word))

        # 查找缓存
        real_prefill_tokens, state = state_cache.check(prefill_tokens)

        # if real_prefill_tokens:
        #     print(f"使用缓存，已省略 {len(prefill_tokens) - len(real_prefill_tokens)} 个 token")

        # 创建 completion 对象
        completion = engine_core.completion(
            prompt_str=prompt,
            prefill_tokens=real_prefill_tokens,
            state=state,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            stop_tokens=set(DEFAULT_STOP_TOKENS + stop_tokens),
            cache_prefill=config.state_cache_size > 0,
            cache_prefill_padding=cache_prefill_padding,
        )

        if request.stream:
            return StreamingResponse(
                stream_chat_completion(completion, request, connection),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                create_non_stream_completion_with_keep_alive(completion, request, prefill_tokens),
                media_type="application/json",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


async def stream_chat_completion(
    completion: AsyncEngineCompletion,
    request: ChatCompletionRequest,
    connection: Request,
) -> AsyncGenerator[str, None]:
    """流式返回聊天完成，包含保活机制"""

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    string_parser = StreamingStringParser(tries=TRIE_THINK_NO_TRIGGER)
    string_parser.parse(completion.task.prompt_str.split("\n\n")[-1])

    combined_stream = asyncio.Queue()
    gen_task = None

    try:

        async def handel_gen_task():
            try:
                async for event in completion:
                    if event[0] == "token":
                        for text, state in string_parser.parse(event[2]):
                            if state == "content":
                                chunk = ChatCompletionStreamResponse(
                                    id=completion_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamChoice(
                                            index=0,
                                            delta={
                                                "content": text,
                                            },
                                        )
                                    ],
                                )
                            elif state == "reasoning_content":
                                chunk = ChatCompletionStreamResponse(
                                    id=completion_id,
                                    created=created,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionStreamChoice(
                                            index=0,
                                            delta={"content": "", "reasoning_content": text},
                                        )
                                    ],
                                )
                            else:
                                continue
                            combined_stream.put_nowait(f"data: {chunk.model_dump_json()}\n\n")
                    elif event[0] == "cache_prefill":
                        state_cache.cache(event[1]["prefilled_tokens"], event[1]["state"])
                        print("已缓存预填充", event[1]["prefilled_tokens"])

            except Exception as e:
                error_chunk = {"error": {"message": str(e), "type": "internal_error"}}
                combined_stream.put_nowait(f"data: {json.dumps(error_chunk)}\n\n")
                combined_stream.put_nowait("data: [DONE]\n\n")
                combined_stream.put_nowait(None)

            final_chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionStreamChoice(index=0, delta={}, finish_reason="stop")],
            )
            combined_stream.put_nowait(f"data: {final_chunk.model_dump_json()}\n\n")
            combined_stream.put_nowait("data: [DONE]\n\n")
            combined_stream.put_nowait(None)

        gen_task = asyncio.create_task(handel_gen_task())

        while True:
            try:
                chunk = await asyncio.wait_for(combined_stream.get(), timeout=10)
                if chunk is None:
                    break
                yield chunk
            except asyncio.TimeoutError:
                yield ":\n\n"

    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "internal_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        if gen_task:
            gen_task.cancel()
        completion.abort()


async def create_non_stream_completion_with_keep_alive(
    completion: AsyncEngineCompletion,
    request: ChatCompletionRequest,
    prefill_tokens: List[int],
) -> AsyncGenerator[str, None]:
    """创建非流式聊天完成响应，包含保活机制"""

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        # 为处理缓存
        async def full_completion_handle():
            result = ""
            async for event in completion:
                if event[0] == "token":
                    result += event[2]
                elif event[0] == "cache_prefill":
                    state_cache.cache(event[1]["prefilled_tokens"], event[1]["state"])
            return result

        # 获取响应结果
        while True:
            try:
                response_text = await asyncio.wait_for(asyncio.create_task(full_completion_handle()), timeout=10)
                break
            except asyncio.TimeoutError:
                yield "\n\n"

        # 计算token数量（简化版本）
        prompt_tokens = len(prefill_tokens)
        completion_tokens = len(completion.task.generated_tokens)
        total_tokens = prompt_tokens + completion_tokens

        # 创建响应对象
        response = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionResponseUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

        # 发送最终的JSON响应
        yield response.model_dump_json()

    except Exception as e:
        error_response = {"error": {"message": str(e), "type": "internal_error"}}
        yield json.dumps(error_response)
    finally:
        completion.abort()


@app.post("/v1/batch/translate")
async def create_translation_batch(request: TranslateRequest):
    if not engine_core:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        return StreamingResponse(
            create_non_stream_translates(request),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


def create_translation_prompt(source_lang, target_lang, text):
    lang_names = {
        "zh-CN": "Chinese",
        "zh-TW": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
    }
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    prompt = f"{source_name}: {text}\n\n{target_name}:"
    return prompt


async def create_non_stream_translates(request: TranslateRequest) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:

        tasks = [
            engine_core.completion(
                create_translation_prompt(request.source_lang, request.target_lang, i), max_tokens=2048, temperature=0.5
            )
            for i in request.text_list
        ]

        translation_task = asyncio.gather(*[task.get_full_completion() for task in tasks])

        while True:
            try:
                response_list = await asyncio.wait_for(translation_task, timeout=10)
                break
            except asyncio.TimeoutError:
                yield "\n\n"

        # 创建响应对象
        response = TranslateResponse(
            translations=[
                TranslationResult(
                    detected_source_lang=request.source_lang,
                    text=i,
                )
                for i in response_list
            ],
            created=created,
            id=completion_id,
        )
        # 发送最终的JSON响应
        yield response.model_dump_json()

    except Exception as e:
        error_response = {"error": {"message": str(e), "type": "internal_error"}}
        yield json.dumps(error_response)
    finally:
        for task in tasks:
            task.abort()


@app.post("/v1/batch/rollout")
async def create_rollout_batch(request: RolloutRequest):
    if not engine_core:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        if request.stream:
            return StreamingResponse(
                create_stream_rollout(request),
                media_type="application/json",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                create_non_stream_rollout(request),
                media_type="application/json",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


async def create_non_stream_rollout(request: RolloutRequest) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:

        tasks = [
            engine_core.completion(
                i,
                state=None,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                stop_tokens=request.stop_tokens,
            )
            for i in request.contents
        ]

        rollout_task = asyncio.gather(*[task.get_full_completion() for task in tasks])

        while True:
            try:
                response_list = await asyncio.wait_for(rollout_task, timeout=10)
                break
            except asyncio.TimeoutError:
                yield "\n\n"

        # 创建响应对象
        response = RolloutStreamResponse(
            rollouts=[
                RolloutStreamChoice(
                    index=i,
                    delta={"content": response_list[i]},
                )
                for i in range(len(response_list))
            ],
            created=created,
            id=completion_id,
        )
        # 发送最终的JSON响应
        yield response.model_dump_json()

    except Exception as e:
        error_response = {"error": {"message": str(e), "type": "internal_error"}}
        yield json.dumps(error_response)
    finally:
        for task in tasks:
            task.abort()


async def create_stream_rollout(
    request: RolloutRequest,
) -> AsyncGenerator[str, None]:
    """流式返回聊天完成，包含保活机制"""

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    combined_stream = asyncio.Queue()
    gen_task = None

    try:

        tasks = [
            engine_core.completion(
                i,
                state=None,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                stop_tokens=request.stop_tokens,
            )
            for i in request.contents
        ]

        async def handel_gen_task(index, completion: AsyncEngineCompletion):
            try:
                async for event in completion:
                    if event[0] == "token":
                        combined_stream.put_nowait((index, event[2]))
            finally:
                combined_stream.put_nowait(None)

        request_count = len(tasks)
        gen_tasks = [asyncio.create_task(handel_gen_task(i, task)) for i, task in enumerate(tasks)]

        while True:
            try:
                chunk = await asyncio.wait_for(combined_stream.get(), timeout=5)
                if chunk is None:
                    request_count -= 1
                    if request_count <= 0:
                        break
                    continue
                else:
                    yield f"data: {RolloutStreamResponse(
                        choices=[
                            RolloutStreamChoice(
                                index=chunk[0],
                                delta={"content": chunk[1]},
                            )
                        ],
                        model="rwkv-latest",
                        created=created,
                        id=completion_id,
                    ).model_dump_json()}\n\n"
            except asyncio.TimeoutError:
                yield ":\n\n"

    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "internal_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        for task in tasks:
            task.abort()


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return ErrorResponse(
        error={
            "message": exc.detail,
            "type": "invalid_request_error",
            "code": exc.status_code,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    return ErrorResponse(error={"message": str(exc), "type": "internal_server_error"})


if __name__ == "__main__":
    # 加载配置
    config = get_config()

    # 启动服务器
    uvicorn.run(app, host=config.host, port=config.port, reload=False, log_level="info")
