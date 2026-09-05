#!/usr/bin/env python3
"""
Benchmark script for Pixal3D (TencentARC)
Đo đạc VRAM đỉnh, thời gian sinh và xuất file 3D GLB
"""

import os
import sys
import time
import argparse
import json
import torch
from PIL import Image

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        return {
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "max_allocated_mb": round(max_allocated, 2),
            "max_reserved_mb": round(max_reserved, 2)
        }
    return {}

def main():
    parser = argparse.ArgumentParser(description="Benchmark Pixal3D")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output_pixal3d.glb", help="Output GLB file")
    parser.add_argument("--low_vram", action="store_true", help="Enable low VRAM mode (CPU offloading)")
    parser.add_argument("--resolution", type=int, default=1024, help="Target resolution (1024 or 1536)")
    args = parser.parse_args()

    print(f"[*] Bắt đầu Benchmark Pixal3D: image={args.image}, low_vram={args.low_vram}, resolution={args.resolution}")
    
    torch.cuda.reset_peak_memory_stats()
    start_load = time.time()
    
    # Run via subprocess or import depending on repository layout
    cmd = [
        sys.executable, "inference.py",
        "--image", args.image,
        "--output", args.output,
    ]
    if args.low_vram:
        cmd.append("--low_vram")

    print(f"[*] Lệnh thực thi: {' '.join(cmd)}")
    load_time = time.time() - start_load

    start_infer = time.time()
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    infer_time = time.time() - start_infer

    print(proc.stdout)
    if proc.returncode != 0:
        print("[!] Lỗi khi chạy inference Pixal3D:", proc.stderr)

    mem_usage = get_vram_usage()
    file_size_kb = round(os.path.getsize(args.output) / 1024, 2) if os.path.exists(args.output) else 0

    result = {
        "model": "Pixal3D",
        "image": args.image,
        "low_vram": args.low_vram,
        "resolution": args.resolution,
        "return_code": proc.returncode,
        "infer_time_s": round(infer_time, 2),
        "vram_peak_mb": mem_usage.get("max_allocated_mb", 0),
        "output_file": args.output,
        "file_size_kb": file_size_kb
    }

    print("\n=== KẾT QUẢ BENCHMARK PIXAL3D (JSON) ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
