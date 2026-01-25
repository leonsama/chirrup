#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键检测本机任意两张 GPU 之间的 P2P 连接速度
并给出 NVLink / PCIe 判断
"""
import subprocess, re, sys, torch, time
from itertools import combinations


# ------------------------------------------------------------------
# 1. 工具函数
# ------------------------------------------------------------------
def run(cmd):
    """Run shell cmd and return stdout str"""
    return subprocess.check_output(cmd, shell=True).decode(errors="ignore")


def gpu_count():
    return torch.cuda.device_count()


def nvlink_status():
    """
    返回 dict:  (i,j) -> ACTIVE 链路条数
    只解析 GPU0 视角，其余卡对称即可
    """
    out = run("nvidia-smi nvlink -s -i 0")
    links = re.findall(r"GPU0\s+-\s+GPU(\d+)\s+:\s+Link\s+\d+:\s+(ACTIVE|INACTIVE)", out)
    # links => [('1', 'ACTIVE'), ('2', 'INACTIVE'), ...]
    active = {}
    for gid, st in links:
        pair = tuple(sorted((0, int(gid))))
        active[pair] = active.get(pair, 0) + (1 if st == "ACTIVE" else 0)
    return active


def p2p_bandwidth(src, dst, size_mb=33.5 * 33, warmup=3):
    """实测单程 P2P 带宽 GB/s"""
    torch.cuda.set_device(src)
    numel = int(size_mb * 1024 * 1024 // 4)

    buf = torch.randn(numel, dtype=torch.float32, device=f"cuda:{src}")
    torch.cuda.synchronize()
    # warmup
    for _ in range(warmup):
        _ = buf.to(f"cuda:{dst}")
    torch.cuda.synchronize()
    # real timing
    tic = time.perf_counter()
    _ = buf.to(f"cuda:{dst}")
    torch.cuda.synchronize()
    toc = time.perf_counter()
    seconds = toc - tic
    gbytes = buf.numel() * 4 / 1e9
    return gbytes / seconds


def classify_link(bw):
    """根据实测带宽给结论"""
    if bw >= 70:
        return "NVLink ✔"
    if bw >= 25:
        return "PCIe 4.0 x16"
    if bw >= 12:
        return "PCIe 3.0 x16"
    return "P2P 未开启 / 降速 ✘"


# ------------------------------------------------------------------
# 2. 主流程
# ------------------------------------------------------------------
def main():
    n = gpu_count()
    if n < 2:
        print("本机 GPU 数量 < 2，无需检测。")
        sys.exit(0)

    print(f"检测到 {n} 张 GPU，开始两两测速 ...\n")
    # 先扫一遍 NVLink 链路（仅 GPU0 视角）
    nvlink_map = nvlink_status()

    header = f"{'Src':>3} → {'Dst':>3} | {'Bandwidth GB/s':>12} | {'Link Type':>15}"
    print(header)
    print("-" * len(header))
    for i, j in combinations(range(n), 2):
        try:
            bw = p2p_bandwidth(i, j)
            link_type = classify_link(bw)
            # 如果 nvidia-smi 显示有 ACTIVE 链路，给出额外提示
            pair = tuple(sorted((i, j)))
            if nvlink_map.get(pair, 0) > 0:
                link_type += f"  ({nvlink_map[pair]}×NVLink active)"
            print(f"{i:>3} → {j:>3} | {bw:>14.1f} | {link_type:>15}")
        except Exception as e:
            print(f'{i:>3} → {j:>3} | {"ERROR":>14} | {str(e):>15}')
    print("-" * len(header))
    for j, i in combinations(range(n), 2):
        try:
            bw = p2p_bandwidth(i, j)
            link_type = classify_link(bw)
            # 如果 nvidia-smi 显示有 ACTIVE 链路，给出额外提示
            pair = tuple(sorted((i, j)))
            if nvlink_map.get(pair, 0) > 0:
                link_type += f"  ({nvlink_map[pair]}×NVLink active)"
            print(f"{i:>3} → {j:>3} | {bw:>14.1f} | {link_type:>15}")
        except Exception as e:
            print(f'{i:>3} → {j:>3} | {"ERROR":>14} | {str(e):>15}')


if __name__ == "__main__":
    main()
