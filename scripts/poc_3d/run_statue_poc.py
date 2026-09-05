#!/usr/bin/env python3
"""
PoC Runner: Chạy FULL luồng Statue Studio (pipeline/statue_pipeline.py)
- Thiết lập dimension nguyên bản cao nhất:
  + Trellis: 1024 grid, 300k decimate target, 4096 texture bake (30 steps)
  + Pixal3D: 1536 grid, 1M decimation target, 4096 texture bake (12 steps)
- Chạy qua toàn bộ các bước tối ưu hóa của 3D Statue:
  1. AI 3D Reconstruction
  2. Clean & repair manifold geometry + Auto ground & orient Y-Up + Pedestal đế tượng
  3. Polygon decimation tối ưu 60fps WebGL/Mobile (50.000 faces target)
  4. Multi-part anatomical / texture color segmentation (sẵn sàng cho bucket fill)
  5. Xuất toàn bộ gói 6 biến thể GLB (plaster, segmented, id_colored, textured, shell, shell_optimized) + statue_package.zip
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.statue_pipeline import Statue3DPipeline

def main():
    parser = argparse.ArgumentParser(description="Chạy Full Luồng 3D Statue Studio PoC")
    parser.add_argument("--input", "-i", type=str, required=True, help="Đường dẫn ảnh đầu vào (.png, .jpg)")
    parser.add_argument("--output_dir", "-o", type=str, default="results/poc_statue_output", help="Thư mục xuất kết quả")
    parser.add_argument("--generator", "-g", type=str, default="trellis", choices=["trellis", "pixal3d"], help="Bộ sinh 3D ('trellis' hoặc 'pixal3d')")
    parser.add_argument("--mesh_detail", type=str, default="high", choices=["preview", "standard", "high"], help="Độ chi tiết mesh (mặc định 'high' - nguyên bản)")
    parser.add_argument("--texture_detail", type=str, default="high", choices=["preview", "standard", "high"], help="Độ chi tiết texture (mặc định 'high' - bake 4096)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    parser.add_argument("--pedestal", type=str, default="round", help="Kiểu đế tượng: round, square, bevel, none")
    args = parser.parse_args()

    input_p = Path(args.input).resolve()
    if not input_p.exists():
        print(f"[!] Lỗi: Không tìm thấy file ảnh '{input_p}'")
        sys.exit(1)

    print("=" * 75)
    print(f"🏛️ BẮT ĐẦU CHẠY FULL PIPELINE 3D STATUE STUDIO ({args.generator.upper()})")
    print(f"Ảnh đầu vào:        {input_p}")
    print(f"Thư mục xuất:       {args.output_dir}")
    print(f"Cấu hình Mesh:      {args.mesh_detail} (Nguyên bản tối đa)")
    print(f"Cấu hình Texture:   {args.texture_detail} (Bake 4096)")
    print(f"Đế tượng:           {args.pedestal}")
    print("=" * 75)

    pipeline = Statue3DPipeline(root_dir=str(ROOT_DIR))
    job_id = f"poc_{args.generator}_{int(time.time())}_{input_p.stem}"

    def on_progress(pct, msg, step_idx, total_steps):
        print(f"[{pct:3d}%] [Step {step_idx}/{total_steps}] {msg}")

    t0 = time.time()
    result = pipeline.process_statue(
        input_path=str(input_p),
        output_dir=args.output_dir,
        job_id=job_id,
        generator_type=args.generator,
        mesh_detail=args.mesh_detail,
        texture_detail=args.texture_detail,
        target_faces=50000,
        pedestal_shape=args.pedestal,
        pedestal_height=0.05,
        target_height=1.6,
        flatten_bottom=True,
        orientation="auto",
        enable_rigging=False,
        seed=args.seed,
        progress_callback=on_progress
    )
    t_total = time.time() - t0

    print("\n" + "=" * 75)
    print("✅ TOÀN BỘ PIPELINE 3D STATUE ĐÃ HOÀN THÀNH XUẤT SẮC!")
    print(f"Tổng thời gian xử lý:       {t_total:.2f}s ({t_total/60:.2f} phút)")
    print(f"Thời gian Stage 0 (Gen 3D):  {result.get('timings', {}).get('generation_sec', 0):.2f}s")
    print(f"Thời gian Tối ưu Statue:    {result.get('timings', {}).get('optimization_sec', 0):.2f}s")
    print("\nDanh sách file sản phẩm đã xuất:")
    for k, v in result.get("artifacts", {}).items():
        if os.path.exists(str(v)):
            sz_mb = os.path.getsize(str(v)) / (1024 * 1024)
            print(f" - {k:<20}: {v} ({sz_mb:.2f} MB)")
    print("=" * 75)

    # Xuất metrics JSON
    metrics = {
        "job_id": job_id,
        "generator": args.generator,
        "input_image": str(input_p),
        "mesh_detail": args.mesh_detail,
        "texture_detail": args.texture_detail,
        "total_time_sec": round(t_total, 2),
        "timings": result.get("timings", {}),
        "artifacts": result.get("artifacts", {}),
        "stats": result.get("stats", {})
    }
    metrics_path = Path(args.output_dir) / job_id / "poc_statue_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Đã lưu file tổng hợp metrics: {metrics_path}")

if __name__ == "__main__":
    main()
