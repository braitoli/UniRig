#!/usr/bin/env python3
import os
import sys
import argparse
import json
import time
from pathlib import Path

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.unirig_pipeline import UniRigPipeline

def main():
    parser = argparse.ArgumentParser(
        description="UniRig 2D Image-to-3D Rigged & Animated Character Pipeline (TRELLIS.2-4B)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input 2D image (.png, .jpg, .jpeg, .webp) or 3D mesh (.glb, .obj)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="results/image_rig_output",
        help="Output directory for generated GLB and pipeline artifacts"
    )
    parser.add_argument(
        "--generator", "-g",
        type=str,
        default="trellis",
        choices=["trellis", "hunyuan3d"],
        help="2D-to-3D AI Generator: 'trellis' (Microsoft TRELLIS.2-4B) or 'hunyuan3d' (Tencent Hunyuan3D-2.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--3d_only",
        action="store_true",
        help="Only generate 3D GLB model from 2D image (Stage 0) without running Rig & Animation"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    gen_name = "Tencent Hunyuan3D-2.1" if args.generator.lower() == "hunyuan3d" else "Microsoft TRELLIS.2-4B"
    print("=" * 70)
    print(f"🚀 UniRig Pipeline (Stage 0: {gen_name} Image-to-3D)")
    print(f"Input file:  {input_path}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Generator:   {args.generator} ({gen_name})")
    print(f"Mode:        {'3D Generation Only' if getattr(args, '3d_only', False) else 'Full Pipeline (3D -> Rig -> Motion)'}")
    print("=" * 70)

    t0 = time.time()
    pipeline = UniRigPipeline(root_dir=str(ROOT_DIR))
    job_id = f"cli_{int(time.time())}_{input_path.stem}"

    if getattr(args, '3d_only', False):
        gen_folder = "stage0_hunyuan3d" if "hunyuan" in args.generator.lower() else "stage0_trellis"
        out_stage0 = Path(args.output_dir) / job_id / gen_folder
        res_0 = pipeline.generate_3d_from_image(
            image_path=str(input_path),
            output_dir=str(out_stage0),
            seed=args.seed,
            generator_type=args.generator
        )
        t1 = time.time()
        print("\n" + "=" * 70)
        print("✅ 3D Model Generation Completed Successfully!")
        print(f"Total time elapsed: {t1 - t0:.2f}s")
        print(f"Model Engine:     {res_0.get('model_used', gen_name)}")
        print(f"Generated 3D GLB: {res_0['output_glb_path']}")
        print(f"Mesh Stats:       {res_0['num_vertices']} vertices, {res_0['num_faces']} faces")
        print("=" * 70)
        return

    res = pipeline.run_full_pipeline(
        input_path=str(input_path),
        job_id=job_id,
        work_dir=args.output_dir,
        generator_type=args.generator,
        seed=args.seed
    )
    t1 = time.time()

    print("\n" + "=" * 70)
    print("✅ Pipeline Completed Successfully!")
    print(f"Total time elapsed: {t1 - t0:.2f}s")
    if res.get("stage0"):
        print(f"Stage 0 (Image -> 3D): {res['stage0']['output_glb_path']} ({res['stage0']['generation_time_sec']}s) [{res['stage0'].get('model_used', gen_name)}]")
    print(f"Stage 1 (Preprocess): {res['prep']['num_vertices']} vertices, {res['prep']['num_faces']} faces")
    print(f"Stage 2 (Skeleton):   {res['skel']['num_bones']} bones predicted ({res['skel']['inference_time_sec']}s)")
    print(f"Stage 3 (Skinning):   {res['skin']['method']} solver ({res['skin']['calc_time_sec']}s)")
    print(f"Stage 4 (Rigged GLB): {res['final_glb']} ({res['rig']['glb_size_bytes'] / 1024 / 1024:.2f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()

