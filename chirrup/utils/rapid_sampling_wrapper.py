"""
Rapid-Sampling 封装模块

此模块封装了 Rapid-Sampling 库的加载和使用，自动检测 CUDA/ROCm 环境并加载相应的实现。
"""

import torch
from pathlib import Path

# Rapid-Sampling 模块（延迟加载）
_rapid_sampling_module = None


def _get_rapid_sampling_path() -> Path:
    """获取 rapid_sampling submodule 的路径"""
    return Path(__file__).parent / "rapid_sampling"


def load_rapid_sampling():
    """
    延迟加载 Rapid-Sampling 模块
    自动检测 CUDA/ROCm 环境并编译相应的实现
    """
    global _rapid_sampling_module
    
    if _rapid_sampling_module is not None:
        return _rapid_sampling_module
    
    rapid_sampling_path = _get_rapid_sampling_path()
    
    if not rapid_sampling_path.exists():
        raise RuntimeError(
            f"Rapid-Sampling submodule not found at {rapid_sampling_path}. "
            "Please run: git submodule update --init --recursive"
        )
    
    from torch.utils.cpp_extension import load
    
    ROCm_flag = torch.version.hip is not None
    
    if ROCm_flag:
        # AMD GPU (ROCm/HIP)
        sources = [
            str(rapid_sampling_path / "hip" / "sampling_op.hip"),
            str(rapid_sampling_path / "hip" / "sampling.hip"),
        ]
        extra_cuda_cflags = ['-fopenmp', '-ffast-math', '-O3', '-munsafe-fp-atomics']
    else:
        # NVIDIA GPU (CUDA)
        sources = [
            str(rapid_sampling_path / "sampling.cpp"),
            str(rapid_sampling_path / "sampling.cu"),
        ]
        extra_cuda_cflags = ["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"]
    
    _rapid_sampling_module = load(
        name="rapid_sampling",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    
    return _rapid_sampling_module


def is_rapid_sampling_available() -> bool:
    """检查 Rapid-Sampling 是否可用"""
    try:
        load_rapid_sampling()
        return True
    except Exception:
        return False
