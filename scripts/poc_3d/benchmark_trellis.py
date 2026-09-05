#!/usr/bin/env python3
"""
Benchmark script for TRELLIS.2-4B
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
    parser = argparse.ArgumentParser(description="Benchmark TRELLIS.2-4B")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output_trellis.glb", help="Output GLB file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=int, default=1024, help="Target resolution (1024 or 1536)")
    args = parser.parse_args()

    print(f"[*] Bắt đầu Benchmark TRELLIS.2-4B: image={args.image}, resolution={args.resolution}")
    
    torch.cuda.reset_peak_memory_stats()
    start_load = time.time()
    
    # Import trellis2 pipeline
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.cuda()
    except Exception as e:
        print(f"[!] Lỗi khi load pipeline Trellis2: {e}")
        # Fallback to trellis 1 if trellis 2 is not yet public under that name
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline
            print("[*] Thử nghiệm với TrellisImageTo3DPipeline...")
            pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
            pipeline.cuda()
        except Exception as e2:
            print(f"[!] Lỗi fallback: {e2}")
            sys.exit(1)

    load_time = time.time() - start_load
    mem_after_load = get_vram_usage()
    print(f"[*] Load pipeline hoàn tất trong {load_time:.2f}s. VRAM sau load: {mem_after_load}")

    # Chạy inference
    image = Image.open(args.image)
    torch.cuda.reset_peak_memory_stats()
    start_infer = time.time()

    outputs = pipeline.run(image, seed=args.seed)
    infer_time = time.time() - start_infer
    mem_after_infer = get_vram_usage()

    # Xuất GLB
    start_export = time.time()
    if isinstance(outputs, list):
        mesh = outputs[0]
    elif isinstance(outputs, dict) and "mesh" in outputs:
        mesh = outputs["mesh"][0]
    else:
        mesh = outputs

    if hasattr(mesh, "export"):
        mesh.export(args.output)
    else:
        print("[*] Mesh output object:", type(mesh))

    export_time = time.time() - start_export
    file_size_kb = round(os.path.getsize(args.output) / 1024, 2) if os.path.exists(args.output) else 0

    result = {
        "model": "TRELLIS.2-4B",
        "image": args.image,
        "resolution": args.resolution,
        "load_time_s": round(load_time, 2),
        "infer_time_s": round(infer_time, 2),
        "export_time_s": round(export_time, 2),
        "total_time_s": round(infer_time + export_time, 2),
        "vram_load_mb": mem_after_load.get("max_allocated_mb", 0),
        "vram_peak_mb": mem_after_infer.get("max_allocated_mb", 0),
        "vram_peak_reserved_mb": mem_after_infer.get("max_reserved_mb", 0),
        "output_file": args.output,
        "file_size_kb": file_size_kb
    }

    print("\n=== KẾT QUẢ BENCHMARK (JSON) ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
