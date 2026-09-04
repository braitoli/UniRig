import os
import sys
import time
import subprocess
import threading
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Union, Dict, Any, Callable

class Pixal3DImageTo3DGenerator:
    """
    Generates high-fidelity 3D meshes (.glb) from 2D images using Tencent ARC's
    Pixal3D: Pixel-Aligned 3D Generation from Images (SIGGRAPH 2026).
    https://github.com/TencentARC/Pixal3D
    """
    def __init__(
        self,
        model_id: str = "TencentARC/Pixal3D",
        device: str = "cuda"
    ):
        self.model_id = model_id
        self.device = device
        self.trellis_python = "/home/braitoli/miniconda/envs/trellis/bin/python"
        self.infer_script = Path(__file__).resolve().parent / "pixal3d_infer.py"
        self.mv_infer_script = Path(__file__).resolve().parent / "pixal3d_mv_infer.py"

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        """Loads and pre-processes input image to square RGBA format."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        else:
            img = image_input

        img = img.convert("RGBA")
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        w, h = img.size
        max_dim = max(w, h)
        square_img = Image.new("RGBA", (max_dim, max_dim), (255, 255, 255, 0))
        square_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        square_img = square_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        return square_img

    def generate_3d_mesh(
        self,
        image_input: Union[str, Path, Image.Image],
        output_glb_path: str,
        seed: int = 42,
        resolution: str = "1024",
        low_vram: bool = False,
        fov: float = -1.0,
        ss_sampling_steps: int = 12,
        shape_slat_sampling_steps: int = 12,
        tex_slat_steps: int = 12,
        decimation_target: int = 300000,
        texture_size: int = 4096,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Converts 2D image into a 3D GLB mesh using Tencent ARC Pixal3D.
        """
        t0 = time.time()
        output_path = Path(output_glb_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def report(pct: int, msg: str, step_idx: int = 1, total_steps: int = 5):
            if progress_callback and callable(progress_callback):
                try:
                    progress_callback(pct, msg, step_idx, total_steps)
                except Exception:
                    pass

        # Step 1: Preprocess input image
        report(10, "Tiền xử lý ảnh 2D (Tách nền Alpha & căn giữa bbox)", 1, 5)
        if isinstance(image_input, Image.Image):
            temp_img_path = output_path.parent / "temp_pixal3d_input.png"
            image_input.save(str(temp_img_path))
            input_file_str = str(temp_img_path)
        else:
            input_file_str = str(image_input)

        mesh_generated = False

        # Step 2: Initialize Pixal3D
        report(25, "Khởi tạo mạng nơ-ron Tencent ARC Pixal3D (SIGGRAPH 2026)...", 2, 5)

        if not Path(self.trellis_python).exists() or not self.infer_script.exists():
            raise RuntimeError(
                f"[Pixal3D] Missing runtime: python={self.trellis_python} "
                f"script={self.infer_script}"
            )

        cmd = [
            self.trellis_python,
            str(self.infer_script),
            f"--image_path={input_file_str}",
            f"--output_path={output_path}",
            f"--seed={seed}",
            f"--resolution={resolution}",
            f"--fov={fov}",
            f"--ss_sampling_steps={ss_sampling_steps}",
            f"--shape_slat_sampling_steps={shape_slat_sampling_steps}",
            f"--tex_slat_steps={tex_slat_steps}",
            f"--decimation_target={decimation_target}",
            f"--texture_size={texture_size}",
        ]
        if low_vram:
            cmd.append("--low_vram")

        env = os.environ.copy()
        env["PATH"] = f"/home/braitoli/miniconda/envs/trellis/bin:{env.get('PATH', '')}"
        env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env["ATTN_BACKEND"] = "flash_attn"

        proc_state = {"done": False, "res": None}

        def run_sub():
            proc_state["res"] = subprocess.run(cmd, capture_output=True, text=True, env=env)
            proc_state["done"] = True

        sub_t = threading.Thread(target=run_sub, daemon=True)
        sub_t.start()

        # Smoothly update progress from 25% to 75%
        t_sub_start = time.time()
        est_duration = 35.0 if low_vram else 45.0
        while not proc_state["done"]:
            elapsed = time.time() - t_sub_start
            ratio = min(1.0, elapsed / est_duration)
            cur_pct = int(25 + 50 * (1.0 - np.exp(-2.2 * ratio)))
            cur_pct = min(75, max(25, cur_pct))
            report(cur_pct, "Đang chạy Pixal3D Pixel-Aligned Transformer Diffusion...", 3, 5)
            time.sleep(0.4)

        sub_t.join()
        res = proc_state["res"]

        if res and res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            report(85, "Sinh lưới bề mặt PBR qua O-Voxel và khử đa giác thừa...", 4, 5)
            print(f"[Pixal3D] Model 3D tạo thành công: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
            mesh_generated = True
        else:
            err_msg = (res.stderr or res.stdout or "Unknown error") if res else "Unknown error"
            print(f"[Pixal3D] Error during inference:\n{err_msg}")
            err_tail = "\n".join(err_msg.strip().splitlines()[-8:])
            raise RuntimeError(f"Pixal3D generation failed: {err_tail}")

        t_end = time.time()
        report(100, "Đã tạo xong model 3D từ ảnh 2D!", 5, 5)

        return {
            "output_glb_path": str(output_path),
            "generation_time_sec": round(t_end - t0, 2),
            "model_used": "TencentARC/Pixal3D",
            "generator_type": "pixal3d",
        }

    def generate_3d_from_multiview(
        self,
        views_dir: str,
        output_glb_path: str,
        seed: int = 42,
        resolution: int = 1024,
        ss_sampling_steps: int = 6,
        shape_slat_sampling_steps: int = 6,
        tex_slat_steps: int = 6,
        decimation_target: int = 100_000,
        texture_size: int = 1024,
        low_vram: bool = False,
        progress_callback: Optional[Callable[[int, str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Converts a multi-view directory (containing transforms.json and 4 views) into a 3D GLB mesh
        using Tencent ARC Pixal3D Multi-View Pipeline.
        """
        t0 = time.time()
        output_path = Path(output_glb_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def report(pct: int, msg: str, step_idx: int = 1, total_steps: int = 5):
            if progress_callback and callable(progress_callback):
                try:
                    progress_callback(pct, msg, step_idx, total_steps)
                except Exception:
                    pass

        report(15, "Đang nạp 4 góc nhìn chuẩn hóa (Front, Right, Back, Left)...", 1, 5)

        if not Path(self.trellis_python).exists() or not self.mv_infer_script.exists():
            raise RuntimeError(
                f"[Pixal3D-MV] Missing runtime: python={self.trellis_python} "
                f"script={self.mv_infer_script}"
            )

        cmd = [
            self.trellis_python,
            str(self.mv_infer_script),
            f"--views_dir={views_dir}",
            f"--output_path={output_path}",
            f"--seed={seed}",
            f"--resolution={resolution}",
            f"--ss_sampling_steps={ss_sampling_steps}",
            f"--shape_slat_sampling_steps={shape_slat_sampling_steps}",
            f"--tex_slat_steps={tex_slat_steps}",
            f"--decimation_target={decimation_target}",
            f"--texture_size={texture_size}",
        ]
        if low_vram:
            cmd.append("--low_vram")

        env = os.environ.copy()
        env["PATH"] = f"/home/braitoli/miniconda/envs/trellis/bin:{env.get('PATH', '')}"
        env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env["ATTN_BACKEND"] = "flash_attn"

        proc_state = {"done": False, "res": None}

        def run_sub():
            proc_state["res"] = subprocess.run(cmd, capture_output=True, text=True, env=env)
            proc_state["done"] = True

        sub_t = threading.Thread(target=run_sub, daemon=True)
        sub_t.start()

        t_sub_start = time.time()
        # Measured on a GB10: 457s for the 1536 cascade at 12 steps/stage with a 1M-face
        # decimation and a 4096 bake. A 50s estimate here pinned the bar at 80% for most
        # of the run, which reads as a hang.
        est_duration = 450.0 if str(resolution) == "1536" else 150.0
        while not proc_state["done"]:
            elapsed = time.time() - t_sub_start
            ratio = min(1.0, elapsed / est_duration)
            cur_pct = int(25 + 55 * (1.0 - np.exp(-2.0 * ratio)))
            cur_pct = min(80, max(25, cur_pct))
            report(cur_pct, "Đang chạy Pixal3D Multi-View Flow Matching 360°...", 3, 5)
            time.sleep(0.4)

        sub_t.join()
        res = proc_state["res"]

        if res and res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
            report(90, "Hoàn tất tái tạo khối 3D 360° từ 4 góc nhìn!", 4, 5)
            print(f"[Pixal3D-MV] Model 3D Multi-view tạo thành công: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            err_msg = (res.stderr or res.stdout or "Unknown error") if res else "Unknown error"
            print(f"[Pixal3D-MV] Error during MV inference:\n{err_msg}")
            err_tail = "\n".join(err_msg.strip().splitlines()[-8:])
            raise RuntimeError(f"Pixal3D Multi-View generation failed: {err_tail}")

        t_end = time.time()
        report(100, "Đã tạo xong model 3D Multi-View hoàn chỉnh!", 5, 5)

        return {
            "output_glb_path": str(output_path),
            "generation_time_sec": round(t_end - t0, 2),
            "model_used": "TencentARC/Pixal3D-MV",
            "generator_type": "pixal3d_mv",
        }

