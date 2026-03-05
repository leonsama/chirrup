# MMLU 测试脚本 - 使用 chirrup 推理引擎
#
# 参考 Albatross benchmark.py 中 MMLU 测试逻辑，使用 chirrup 引擎进行 MMLU 评估。
# 核心逻辑：对每个 MMLU 问题，forward 整个 prompt，取最后位置的 logits，
# 比较 A/B/C/D 四个选项 token 的 log probability，判断模型选择是否正确。
#
# Usage:
#   python scripts/test_mmlu/test_mmlu.py --model_path ../models/rwkv7-g1c-7.2b-20251231-ctx8192.pth
#   python scripts/test_mmlu/test_mmlu.py --model_path ../models/... --max_samples 100 --show_subject

import argparse
import asyncio
from pathlib import Path
from collections import defaultdict

import torch
from torch.nn import functional as F
from tqdm import tqdm

from datasets import load_from_disk

from chirrup.engine_core import AsyncEngineCore
from chirrup.core_structure import ModelLoadConfig


# MMLU prompt 模板 - 与 Albatross benchmark.py 一致
TEMPLATE = (
    "User: You are a very talented expert in <SUBJECT>. Answer this question:\n"
    "<Q>\n"
    "A. <|A|>\n"
    "B. <|B|>\n"
    "C. <|C|>\n"
    "D. <|D|>\n"
    "\n"
    "Assistant: The answer is"
)

CHOICES = [" A", " B", " C", " D"]


async def eval_single_sample(
    engine_core: AsyncEngineCore,
    prompt_text: str,
    gt: int,
    choices_token: list[int],
) -> tuple[bool, int]:
    """评估单个 MMLU 样本

    Returns:
        (is_correct, predicted_answer_idx)
    """
    # 编码 prompt，前置 token 0（与 Albatross 一致）
    prefix_ids = [0] + engine_core.tokenizer.encode(prompt_text.replace("\r\n", "\n").strip())

    # 使用 chirrup completion，max_tokens=1 只生成一个 token 以获取 logits
    completion = engine_core.completion(
        prompt_str=prompt_text,
        prefill_tokens=prefix_ids,
        state=None,
        max_tokens=1,
        temperature=1.0,
        top_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        penalty_decay=0.996,
        stop_tokens=[],
        return_logits=True,
    )

    logits = None
    async for event in completion:
        if event[0] == "token":
            # event = ("token", token_id, text, logits_tensor) when return_logits=True
            if len(event) > 3 and event[3] is not None:
                logits = event[3]
            break  # 只需要第一个 token 的 logits

    if logits is None:
        return False, -1

    # 计算 log_softmax 并比较四个选项的概率
    log_prob = F.log_softmax(logits.float(), dim=-1)
    target_prob = log_prob[choices_token]
    predicted = torch.argmax(target_prob).item()

    return predicted == gt, predicted


async def run_mmlu_eval(args):
    """运行 MMLU 评估"""

    # 加载数据集
    dataset_path = Path(args.dataset_path)
    assert dataset_path.exists(), f"数据集路径不存在: {dataset_path}"
    print(f"从本地加载数据集: {dataset_path}")
    mmlu_test = load_from_disk(str(dataset_path))
    print(f"MMLU 数据集大小: {len(mmlu_test)} 条")

    # 初始化引擎
    model_config = ModelLoadConfig(
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        vocab_size=65536,
        head_size=64,
    )

    engine_core = AsyncEngineCore()
    print("正在加载模型...")
    await engine_core.init(
        worker_num=args.worker_num,
        model_config=model_config,
        batch_size=args.batch_size,
    )
    print("模型加载完成")

    tokenizer = engine_core.tokenizer

    # 验证选项 token 编码 - 每个选项应编码为单个 token
    choices_token = [tokenizer.encode(x) for x in CHOICES]
    assert all(len(x) == 1 for x in choices_token), (
        f"选项 token 编码长度不为 1: {choices_token}"
    )
    choices_token = [x[0] for x in choices_token]
    print(f"选项 token ids: {dict(zip(CHOICES, choices_token))}")

    # 构建所有 prompt
    samples = []
    for idx, sample in enumerate(mmlu_test):
        question = sample["question"]
        choices = sample["choices"]
        subject = sample["subject"]
        gt = sample["answer"]

        prompt_text = (
            TEMPLATE.replace("<Q>", question)
            .replace("<|A|>", choices[0])
            .replace("<|B|>", choices[1])
            .replace("<|C|>", choices[2])
            .replace("<|D|>", choices[3])
            .replace("<SUBJECT>", subject.replace("_", " "))
        )

        if idx == 0:
            print(f"\nPrompt 示例:")
            print("-" * 80)
            print(prompt_text)
            print("-" * 80)

        samples.append((prompt_text, gt, subject))

    # 限制样本数
    if args.max_samples and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
        print(f"限制评估样本数: {args.max_samples}")

    # 并发评估
    correct = 0
    total = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    pbar = tqdm(total=len(samples), desc="MMLU Eval")

    async def eval_and_track(idx, prompt_text, gt, subject):
        nonlocal correct, total
        is_correct, predicted = await eval_single_sample(
            engine_core, prompt_text, gt, choices_token
        )
        total += 1
        if is_correct:
            correct += 1
        subject_stats[subject]["total"] += 1
        if is_correct:
            subject_stats[subject]["correct"] += 1

        pbar.set_description(
            f"Correct: {correct} / Total: {total} - Accuracy: {correct / total:.4f}"
        )
        pbar.update(1)

    tasks = [
        asyncio.create_task(eval_and_track(idx, prompt_text, gt, subject))
        for idx, (prompt_text, gt, subject) in enumerate(samples)
    ]

    await asyncio.gather(*tasks)
    pbar.close()

    # 输出总结
    print(f"\n{'=' * 80}")
    print(f"MMLU 评估结果")
    print(f"{'=' * 80}")
    print(f"模型: {args.model_path}")
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"总体准确率: {correct / total:.4f} ({correct / total * 100:.2f}%)")

    if args.show_subject:
        print(f"\n{'=' * 80}")
        print(f"分科目准确率:")
        print(f"{'=' * 80}")
        sorted_subjects = sorted(subject_stats.items(), key=lambda x: x[0])
        for subject, stats in sorted_subjects:
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {subject.replace('_', ' '):<50s} {stats['correct']:>4d}/{stats['total']:<4d} = {acc:.4f}")

    engine_core.shutdown()
    print("\n评估完成。")


def main():
    parser = argparse.ArgumentParser(description="MMLU 评估 - chirrup 推理引擎")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型文件路径",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="./Albatross/rwkv_vocab_v20230424.txt",
        help="词表文件路径 (default: ./Albatross/rwkv_vocab_v20230424.txt)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(Path(__file__).parent / "mmlu_test_dataset"),
        help="MMLU 数据集路径 (default: scripts/test_mmlu/mmlu_test_dataset)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Worker batch size (default: 64)",
    )
    parser.add_argument(
        "--worker_num",
        type=int,
        default=1,
        help="Worker 数量 (default: 1)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评估样本数 (default: None, 评估全部)",
    )
    parser.add_argument(
        "--show_subject",
        action="store_true",
        default=False,
        help="是否显示分科目准确率",
    )

    args = parser.parse_args()
    asyncio.run(run_mmlu_eval(args))


if __name__ == "__main__":
    main()
