import asyncio
from openai import AsyncOpenAI
from collections import deque

import time, datetime

DEFAULT_API_URL = "http://localhost:8000/v1"

SYSTEM = """The current time is: {date}.

You are the RWKV large language model (LLM).

RWKV (pronounced RwaKuv) is an RNN with great LLM performance and parallelizable like a Transformer. Check RWKV-7 "Goose" reasoning models.

It's combining the best of RNN and transformer - great performance, linear time, constant space (no kv-cache), fast training, infinite ctxlen, and free text embedding. And it's 100% attention-free."""


async def stream_openai_request():
    client = AsyncOpenAI(base_url=DEFAULT_API_URL, api_key="xxx")
    system = SYSTEM.format(date=datetime.datetime.now().strftime("%Y/%m/%d, %A"))

    messages = [{"role": "system", "content": system}]
    tps_recorder = deque(maxlen=20)

    while (line := input(">>>")) != "":
        messages.append({"role": "user", "content": line})
        stream = await client.chat.completions.create(
            model="rwkv-latest",
            messages=messages,
            stream=True,
            top_p=0.8,
            extra_body={
                # "use_state_cache":False
                # "cache_prefill": False,
            }
        )
        c_start = time.perf_counter()
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                tps_recorder.append(1 / (time.perf_counter() - c_start))
                c_start = time.perf_counter()
        print(f"\n(tps: {sum(tps_recorder)/len(tps_recorder):.2f})")


if __name__ == "__main__":
    asyncio.run(stream_openai_request())
