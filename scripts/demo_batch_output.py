import argparse, asyncio
from tqdm import tqdm

from chirrup.engine_core import AsyncEngineCore
from chirrup.core_structure import ModelLoadConfig


async def batch(model_path: str, batch_size: int, task_num: int, worker_num: int):
    model_config = ModelLoadConfig(
        model_path=model_path,
        vocab_path="../Albatross/reference/rwkv_vocab_v20230424.txt",
        vocab_size=65536,
        head_size=64,
    )

    engine_core = AsyncEngineCore()
    await engine_core.init(worker_num=worker_num, model_config=model_config, batch_size=batch_size + 1)

    pbar = tqdm(total=task_num, unit="Sequence")
    prompts = [f"User: 为什么 {i} 是一个有趣的数字？\n\nAssistant: <think>\n</think>" for i in range(task_num)]

    async def completion_task(k, prompt):
        result = await engine_core.completion(prompt).get_full_completion()
        print(f"\n\n[{k}] >>>{prompt}{result}\n\n")
        pbar.update(1)
        return result

    async def logger():
        worker_output_queue = engine_core.worker_pubsub.sub("worker_0")
        prev_mem_alloc = -1
        while True:
            log = await worker_output_queue.get()
            pbar.set_description(
                f"loop HZ: {(1/log[1]['avg_loop_time']):.3f} | {' '.join([f'{k}:{v}' for k,v in log[1]['task_details'].items()])} | delta mem: {log[1]['max_allocated_memory_GB'] - prev_mem_alloc:.3f} GB"
            )
            prev_mem_alloc = log[1]["max_allocated_memory_GB"]

    logger_task = asyncio.create_task(logger())

    results = await asyncio.gather(
        *[
            completion_task(
                k,
                prompt,
            )
            for k, prompt in enumerate(prompts)
        ]
    )
    engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Batch rollout demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/rwkv7-g0a3-7.2b-20251029-ctx8192.pth",
        help="Path to the model file",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--task_num", type=int, default=100, help="Number of tasks (default: 100)")
    parser.add_argument("--worker_num", type=int, default=1, help="Number of workers (default: 1)")

    args = parser.parse_args()

    asyncio.run(batch(args.model_path, args.batch_size, args.task_num, args.worker_num))


if __name__ == "__main__":
    main()
