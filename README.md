<div align="center">
  <img src="assets/chirrup_logo.png" alt="Chirrup Logo" width="450">
</div>

# Chirrup

> /ˈCHirəp/ — (especially of a small bird) make repeated short high-pitched sounds; twitter.

**Chirrup** is a high-performance inference frontend for RWKV models, built on top of [Albatross](https://github.com/BlinkDL/Albatross).

---

## 📊 Performance

### November 12, 2025

| GPU Configuration   | Model | Workers | BSZ/Worker | Total Concurrent Requests | TPS per Request |
| ------------------- | ----- | ------- | ---------- | ------------------------- | --------------- |
| 4 × RTX 4090 24GB   | 7.2B  | 4       | 200        | 800                       | 16 tps          |
| 4 × Tesla V100 16GB | 7.2B  | 4       | 34         | 136                       | 17 tps          |

> **Note**: The RTX 4090 configuration is far from the GPU's processing limits, with significant optimization potential remaining.

## ✨ Features

### ✅ Implemented

- **High Performance**: Leverages the blazing-fast inference engine from [Albatross](https://github.com/BlinkDL/Albatross).
- **Continuous Batching**: Maximizes GPU utilization by dynamically batching incoming requests.
- **State Cache**: Reuses computation states for long-context inputs, significantly improving throughput as context length increases.
- **OpenAI-Compatible API**: Drop-in replacement for existing LLM workflows — no code changes needed.

### 🔜 Planned

- [ ] CUDA Graph support for reduced kernel launch overhead
- [ ] Prefill-Decode separation for optimized scheduling
- [ ] Constrained decoding (e.g., JSON schema)
- [ ] Function Calling support
- [ ] Pipeline parallelism to enable inference of even larger models

---

## 🚀 Getting Started

### 1. Download a Model

Visit the official model hub and download a RWKV-7 `g1` series model that fits your needs:  
👉 [https://huggingface.co/BlinkDL/rwkv7-g1/tree/main](https://huggingface.co/BlinkDL/rwkv7-g1/tree/main)

### 2. Set Up Environment

For **best performance**, we strongly recommend using **Python 3.14t (Free threading)** via `uv`.

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/leonsama/chirrup.git

# Create a Python 3.14t virtual environment
uv venv --python 3.14t

# Activate it
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate     # Windows

# Install Chirrup
uv pip install -e .

# Install dependencies with CUDA 12.9 support and dev tools
uv sync --extra torch-cu129 --extra dev
```

> 💡 You may use `torch-cu126` or `torch-rocm` instead if your system requires it, or customize the PyTorch backend in `pyproject.toml`.

---

## 🌐 Start API Service

### Quick Start

```bash
# Currently, `triton._C.libtriton` doesn't declare itself GIL-safe, but it actually works fine—so we
#     manually disable the GIL with `PYTHON_GIL=0`.
PYTHON_GIL=0 uv run python -m chirrup.web_service.app --model_path /path/to/your/model
```

The service will start at **`http://127.0.0.1:8000`**, providing OpenAI-compatible API endpoints.

📖 **Detailed Documentation**: Check [./Docs/API.md](./Docs/API.md) for complete command-line parameters and API interface documentation.

---

## 🧪 Run Demos

### Stream Output (Single Request)

[**Demo:**](./test/demo_stream_output.py)

```bash
PYTHON_GIL=0 uv run test/demo_stream_output.py --model_path /path/to/your/model
```

**Code Example:**

```python
from chirrup.engine_core import AsyncEngineCore
from chirrup.core_structure import ModelLoadConfig

model_config = ModelLoadConfig(
    model_path=model_path,
    vocab_path="../Albatross/reference/rwkv_vocab_v20230424.txt",
    vocab_size=65536,
    head_size=64,
)

engine_core = AsyncEngineCore()
await engine_core.init(worker_num=1, model_config=model_config, batch_size=4)

prompt = "User: 为什么 42 是一个有趣的数字？\n\nAssistant:"
completion = engine_core.completion(prompt)

print(prompt, end="", flush=True)
async for event in completion:
    if event[0] == "token":
        print(event[2], end="", flush=True)
```

### Batch Inference (Concurrent Requests)

[**Demo:**](./test/demo_batch_output.py)

```bash
PPYTHON_GIL=0 v run test/demo_batch_output.py --model_path /path/to/your/model --batch_size 32 --task_num 512 --worker_num 4
```

**Code Example:**

```python
from chirrup.engine_core import AsyncEngineCore
from chirrup.core_structure import ModelLoadConfig
import asyncio

model_config = ModelLoadConfig(
    model_path=model_path,
    vocab_path="../Albatross/reference/rwkv_vocab_v20230424.txt",
    vocab_size=65536,
    head_size=64,
)

engine_core = AsyncEngineCore()
await engine_core.init(worker_num=4, model_config=model_config, batch_size=33)  # batch_size = max_batch + 1

prompts = [f"User: 为什么 {i} 是一个有趣的数字？\n\nAssistant: <think>\n</think>" for i in range(512)]

results = await asyncio.gather(
    *[engine_core.completion(prompt).get_full_completion() for prompt in prompts]
)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Thanks to [**RWKV-Vibe/rwkv_lightning**](https://github.com/RWKV-Vibe/rwkv_lightning) for inspiration and to its author **Alic** for valuable guidance.
- Thanks to **Jellyfish** for the [**continuous batching implementation**](https://github.com/BlinkDL/Albatross/pull/5) in Albatross.

---

<div align="center">
🐦 Like a chirping bird — lightweight, fast, and always responsive.
</div>

<div align="center">
<sub>Built with ❤️ for the RWKV ecosystem</sub>
</div>
