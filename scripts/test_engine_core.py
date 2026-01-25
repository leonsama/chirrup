import asyncio

from chirrup.engine_core import AsyncEngineCore
from chirrup.core_structure import ModelLoadConfig
from chirrup.utils.prompt_formatters import format_openai_message_quick_thinking


async def test_engine_core():
    """测试 AsyncEngineCore 的基本功能"""

    print("开始测试 AsyncEngineCore...")

    # 创建模型配置
    model_config = ModelLoadConfig(
        # model_path="../models/rwkv7-g1a3-1.5b-20251015-ctx8192",
        model_path="../models/rwkv7-g1a-0.1b-20250728-ctx4096.pth",
        vocab_path="./Albatross/rwkv_vocab_v20230424.txt",
        vocab_size=65536,
        head_size=64,
    )

    # 创建引擎核心
    engine_core = AsyncEngineCore()

    try:
        # 测试初始化 Worker
        print("测试初始化 Worker...")
        await engine_core.init(worker_num=1, model_config=model_config, batch_size=4)

        print("✓ Worker 初始化成功")

        prompt = format_openai_message_quick_thinking([{"role": "user", "content": "为什么妈妈那天会被吃掉？"}])

        print("测试创建 completion 对象...")
        completion = engine_core.completion(
            prompt_str=prompt,
            prefill_tokens=engine_core.tokenizer.encode(prompt),
            state=None,
        )
        completion1 = engine_core.completion(
            prompt_str=prompt,
            prefill_tokens=engine_core.tokenizer.encode(prompt),
            state=None,
        )
        full_completion = completion1.get_full_completion()

        # 测试任务中断
        print("测试任务中断...")

        async def process_task():
            # 测试创建 completion 对象
            print(f">>> {prompt}", end="", flush="")
            async for event in completion:
                if event[0] == "token":
                    print(event[2], end="", flush=True)
            print("\n", completion.task.request_status)

        gen_task = asyncio.create_task(process_task())
        await asyncio.sleep(5)
        completion.abort()
        await asyncio.sleep(1)
        print("✓ 任务中断信号发送成功")

        print("✓ Completion 对象创建成功")
        print(f"  Task ID: {completion1.task_id}")

        async def logger():
            worker_output_queue = engine_core.worker_pubsub.sub("worker_0")
            while True:
                log = await worker_output_queue.get()
                print(
                    f"loop HZ: {(1/log[1]["avg_loop_time"]):.3f} | {' '.join([f'{k}:{v}' for k,v in log[1]['task_details'].items()])}"
                )

        log_task = asyncio.create_task(logger())

        print("测试一次性获取 completion ...")
        print(f">>> {prompt}", end="", flush="")
        print(await full_completion)
        print(f"✓ 一次性获取 completion 成功")

        print("所有测试通过！")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 清理资源
        print("清理资源...")
        engine_core.shutdown()
        log_task.cancel()
        print("✓ 资源清理完成")


if __name__ == "__main__":
    asyncio.run(test_engine_core())
