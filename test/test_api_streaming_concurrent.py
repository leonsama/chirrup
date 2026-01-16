#!/usr/bin/env python3
"""
并发流式 LLM 测试器 —— 单请求实时 TPOT 队列 + 0.1 s 刷新
用法:
    python llm_concurrent_test.py -c 64
"""
import argparse
import asyncio
import os
import statistics
import time
from collections import deque
from typing import Deque

import tqdm
from openai import AsyncOpenAI


# -------------------- 参数 --------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--concurrency", type=int, default=64,
                        help="并发请求数（默认 64）")
    parser.add_argument("--maxlen", type=int, default=5,
                        help="每个请求计算 TPOT 时保留最近多少个 delta（默认 5）")
    return parser.parse_args()


args = parse_args()
N: int = args.concurrency
MAXLEN: int = args.maxlen

# 无界队列：单请求把当前 TPOT 放进来
tpot_queue: asyncio.Queue[float] = asyncio.Queue()


# -------------------- 单请求协程 --------------------
async def one_stream(request_id: int, client: AsyncOpenAI, pbar: tqdm.tqdm):
    """发一条流式请求，计算本请求当前 TPOT 并塞进队列。"""
    deltas: Deque[float] = deque(maxlen=MAXLEN)
    msg = "为什么 0 是一个有趣的数字？"
    last = time.perf_counter()

    try:
        stream = await client.chat.completions.create(
            model="rwkv-latest",
            messages=[{"role": "user", "content": msg}],
            # max_tokens=150,
            stream=True,
        )
        response = ""
        async for chunk in stream:
            now = time.perf_counter()
            deltas.append(now - last)
            last = now
            # 每收到一个 chunk 就计算一次当前 TPOT 并扔进队列
            if deltas:
                await tpot_queue.put(statistics.mean(deltas))
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        pbar.update(1)
        print(f"[{request_id}] 请求完成，内容: >>>{msg}{response}\n\n")
    except Exception as exc:
        pbar.write(f"Request {request_id} error: {exc}")
        pbar.update(1)


# -------------------- 后台 TPOT 刷新器 --------------------
async def tpot_reporter(pbar: tqdm.tqdm):
    """每 0.1 s 一次性取空队列，直接平均后更新进度条后缀。"""
    while True:
        await asyncio.sleep(0.1)
        bucket = []
        # 一次性取空
        while not tpot_queue.empty():
            bucket.append(tpot_queue.get_nowait())
        if bucket:
            avg = statistics.mean(bucket)
            pbar.set_postfix({"TPS": f"{(1/avg):.3f}"})


# -------------------- 主入口 --------------------
async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="test",
    )

    with tqdm.tqdm(total=N, unit="req", desc="Streaming") as pbar:
        # 启动后台 reporter
        reporter = asyncio.create_task(tpot_reporter(pbar))

        # 并发请求
        await asyncio.gather(*(one_stream(i + 1, client, pbar) for i in range(N)))

        # 全部完成后，让 reporter 再刷一次并优雅结束
        await asyncio.sleep(0.2)
        reporter.cancel()

    print("\n全部请求完成。")


if __name__ == "__main__":
    asyncio.run(main())