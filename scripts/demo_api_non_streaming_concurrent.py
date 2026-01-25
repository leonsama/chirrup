import asyncio
import time
from openai import AsyncOpenAI

async def send_request(client: AsyncOpenAI, model, messages, request_id):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=100,   
        )
        content = response.choices[0].message.content
        print(f"Request {request_id}: {content}")
        return content
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return None

async def main():
    # 创建 OpenAI 异步客户端
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",  # 设置基础 URL
        api_key="sk-test-key"  # API 密钥 (本地部署通常不需要验证)
    )

    # 创建任务列表
    tasks = []
    for i in range(0, 20):
        messages = [{"role": "user", "content": f"为什么 0 是一个有趣的数字？"}]
        task = send_request(client, "rwkv-latest", messages, i)
        tasks.append(task)

    # 并发执行所有请求
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    # 统计结果
    successful_requests = sum(1 for result in results if result is not None)
    print(f"\nCompleted {successful_requests}/{len(tasks)} requests successfully")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())