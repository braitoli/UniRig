#!/usr/bin/env python3
"""
Batch Statue Studio Benchmark Runner
Chạy đo đạc tự động chuỗi ảnh đại diện đa dạng thể loại cho TRELLIS 2 (Warm Start)
và Pixal3D (Low-VRAM / Standard) trên máy chủ RTX 4090.
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

from scripts.poc_3d.statue_detailed_benchmark import run_statue_detailed_benchmark

BENCHMARK_IMAGES = [
    {"name": "cat.png", "category": "Sinh vật quen thuộc / đối chứng GX10"},
    {"name": "sample_character.png", "category": "Nhân vật hoạt hình anime/game"},
    {"name": "typical_creature_quadruped.png", "category": "Thú 4 chân"},
    {"name": "typical_creature_dragon.png", "category": "Linh thú phức tạp / rồng"},
    {"name": "typical_creature_robot_crab.png", "category": "Cua robot / chi tiết cứng"},
    {"name": "typical_humanoid_mech.png", "category": "Robot người / khớp nối"},
    {"name": "typical_humanoid_dwarf.png", "category": "Nhân vật dáng người thấp"},
    {"name": "typical_building_mushroom.png", "category": "Nhà nấm / kiến trúc organic"},
    {"name": "typical_building_castle.png", "category": "Lâu đài kiến trúc góc cạnh"},
    {"name": "typical_vehicle_locomotive.png", "category": "Xe lửa / phương tiện"},
    {"name": "typical_misc_monster_chest.png", "category": "Rương quái vật / đồ vật kết hợp sinh vật"},
    {"name": "typical_misc_lantern.png", "category": "Đèn lồng / vật thể chi tiết"},
]

def run_batch(
    generator_type="trellis",
    mesh_detail="high",
    texture_detail="high",
    target_faces=0,
    max_images=None,
    output_dir="results/statue_batch_benchmark"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = ROOT_DIR / "scripts" / "poc_3d" / "samples"
    
    images_to_run = BENCHMARK_IMAGES[:max_images] if max_images else BENCHMARK_IMAGES
    all_results = []
    
    print("=" * 90)
    print(f"🚀 BẮT ĐẦU CHẠY BATCH BENCHMARK STATUE STUDIO ({len(images_to_run)} ẢNH)")
    print(f"Mô hình:         {generator_type.upper()}")
    print(f"Mesh Detail:     {mesh_detail} (Nguyên bản 100%, target_faces={target_faces})")
    print(f"Texture Detail:  {texture_detail} (Bake 4096)")
    print(f"Thư mục lưu:     {out_dir}")
    print("=" * 90)
    
    t_batch_start = time.time()
    
    for idx, item in enumerate(images_to_run, start=1):
        img_name = item["name"]
        category = item["category"]
        img_path = samples_dir / img_name
        
        if not img_path.exists():
            print(f"[-] Bỏ qua {img_name}: Không tìm thấy file tại {img_path}")
            continue
            
        print(f"\n[{idx}/{len(images_to_run)}] 👉 TIẾN HÀNH TEST: {img_name} ({category})")
        t_img_start = time.time()
        
        try:
            res = run_statue_detailed_benchmark(
                image_path=str(img_path),
                output_dir=str(out_dir),
                generator_type=generator_type,
                mesh_detail=mesh_detail,
                texture_detail=texture_detail,
                target_faces=target_faces
            )
            res["category"] = category
            res["image_name"] = img_name
            all_results.append(res)
            print(f"[✓] Hoàn tất {img_name} trong {time.time() - t_img_start:.2f}s!")
        except Exception as e:
            print(f"[✗] Thất bại tại {img_name}: {e}")
            all_results.append({
                "image_name": img_name,
                "category": category,
                "error": str(e)
            })
            
    total_batch_time = round(time.time() - t_batch_start, 2)
    
    # Save consolidated summary JSON
    summary_path = out_dir / f"batch_summary_{generator_type}_{int(time.time())}.json"
    summary_data = {
        "generator_type": generator_type,
        "mesh_detail": mesh_detail,
        "texture_detail": texture_detail,
        "target_faces": target_faces,
        "total_batch_time_sec": total_batch_time,
        "num_images": len(all_results),
        "successful_images": sum(1 for r in all_results if "summary" in r),
        "results": all_results
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
        
    print("\n" + "=" * 90)
    print(f"🏁 HOÀN TẤT TOÀN BỘ BATCH ({len(all_results)} ảnh) TRONG {total_batch_time}s ({total_batch_time/60:.2f} phút)!")
    print(f"File tổng hợp kết quả: {summary_path}")
    print("=" * 90)
    return summary_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Statue Studio Benchmark")
    parser.add_argument("--generator", "-g", type=str, default="trellis", choices=["trellis", "pixal3d"])
    parser.add_argument("--mesh_detail", type=str, default="high")
    parser.add_argument("--texture_detail", type=str, default="high")
    parser.add_argument("--target_faces", type=int, default=0)
    parser.add_argument("--max_images", "-n", type=int, default=None)
    parser.add_argument("--output_dir", "-o", type=str, default="results/statue_batch_benchmark")
    args = parser.parse_args()
    
    run_batch(
        generator_type=args.generator,
        mesh_detail=args.mesh_detail,
        texture_detail=args.texture_detail,
        target_faces=args.target_faces,
        max_images=args.max_images,
        output_dir=args.output_dir
    )
