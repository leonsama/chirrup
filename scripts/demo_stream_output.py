import argparse, asyncio

from chirrup.engine_core import AsyncEngineCore
from chirrup.core_structure import ModelLoadConfig


async def main(model_path):
    model_config = ModelLoadConfig(
        model_path=model_path,
        vocab_path="../Albatross/reference/rwkv_vocab_v20230424.txt",
        vocab_size=65536,
        head_size=64,
    )

    # 创建引擎核心
    engine_core = AsyncEngineCore()
    await engine_core.init(worker_num=1, model_config=model_config, batch_size=4)

    prompt = "User: 为什么 42 是一个有趣的数字？\n\nAssistant:"

    completion = engine_core.completion(prompt)

    print(prompt, end="", flush=True)

    async for event in completion:
        if event[0] == "token":
            print(event[2], end="", flush=True)
    print()
    engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch rollout demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/rwkv7-g0a3-7.2b-20251029-ctx8192.pth",
        help="Path to the model file",
    )
    args = parser.parse_args()

    asyncio.run(main(args.model_path))
