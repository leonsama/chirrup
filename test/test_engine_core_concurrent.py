import asyncio
from tqdm import tqdm

from chirrup.engine_core import AsyncEngineCore
from chirrup.core_structure import ModelLoadConfig


async def test_engine_core():
    """测试 AsyncEngineCore 的基本功能"""

    # 创建模型配置
    model_config = ModelLoadConfig(
        model_path="../models/rwkv7-g1c-1.5b-20260110-ctx8192",
        # model_path="../models/rwkv7-g1c-7.2b-20251231-ctx8192.pth",
        vocab_path="./Albatross/rwkv_vocab_v20230424.txt",
        vocab_size=65536,
        head_size=64,
    )

    # 创建引擎核心
    engine_core = AsyncEngineCore()

    try:
        # 测试初始化 Worker
        print("测试初始化 Worker...")
        await engine_core.init(worker_num=1, model_config=model_config, batch_size=80 + 1)

        print("测试创建 completion 对象...")

        total = 256
        pbar = tqdm(total=total, unit="Sequence")

        prompts = [f"User: 为什么 {i} 是一个有趣的数字？\n\nAssistant: <think>" for i in range(total)]

        async def create_full_completion(prompt, k):
            result = await engine_core.completion(prompt, cache_prefill=True).get_full_completion()
            print(f"\n\n[{k}] >>>{prompt}{result}\n\n")
            # print(prompt)
            pbar.update(1)

        async def logger():
            prev_mem_alloc = -1
            async for perf in engine_core.iter_worker_performance():
                if perf["worker_id"] == "worker_0":
                    pbar.set_description(
                        f"loop HZ: {(1/perf['avg_loop_time']):.3f} | "
                        f"{' '.join([f'{k}:{v}' for k, v in perf['task_details'].items()])} | "
                        f"delta mem: {perf['max_allocated_memory_GB'] - prev_mem_alloc:.3f} GB"
                    )
                    prev_mem_alloc = perf["max_allocated_memory_GB"]

        log_task = asyncio.create_task(logger())

        print("启动 completion ...")
        results = await asyncio.gather(
            *[
                create_full_completion(
                    prompt,
                    k
                )
                for k, prompt in enumerate(prompts)
            ]
        )

        # print(results)

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 清理资源
        pbar.close()
        print("清理资源...")
        engine_core.shutdown()
        print("✓ 资源清理完成")


if __name__ == "__main__":
    # import yappi

    # yappi.start()

    asyncio.run(test_engine_core())

    # yappi.stop()

    # retrieve thread stats by their thread id (given by yappi)
    # threads = yappi.get_thread_stats()
    # for thread in threads:
    #     print("Function stats for (%s) (%d)" % (thread.name, thread.id))  # it is the Thread.__class__.__name__
    #     stats = yappi.get_func_stats(ctx_id=thread.id)
    #     with open(f"thread_stats_{thread.id}.txt", "w") as f:
    #         stats.print_all(out=f)
