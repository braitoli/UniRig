import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Union, Dict, Any
import trimesh

class TrellisImageTo3DGenerator:
    """
    Generates 3D meshes (.glb) from 2D images using Microsoft's TRELLIS.2-4B model
    (https://huggingface.co/microsoft/TRELLIS.2-4B).
    """
    def __init__(
        self,
        model_id: str = "microsoft/TRELLIS.2-4B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype
        self.pipeline = None
        self._is_loaded = False

    def load_pipeline(self):
        """Lazily load the TRELLIS.2 pipeline from Hugging Face."""
        if self._is_loaded and self.pipeline is not None:
            return

        print(f"[TrellisGenerator] Loading {self.model_id} on {self.device}...")
        t0 = time.time()
        
        try:
            # Try importing trellis2 or trellis pipeline module if available
            try:
                from trellis2.pipelines import Trellis2ImageTo3DPipeline
                self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype
                )
                self.pipeline.to(self.device)
                self._is_loaded = True
                print(f"[TrellisGenerator] Trellis2ImageTo3DPipeline loaded in {time.time() - t0:.2f}s")
                return
            except ImportError:
                pass

            try:
                from trellis.pipelines import TrellisImageTo3DPipeline
                self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype
                )
                self.pipeline.to(self.device)
                self._is_loaded = True
                print(f"[TrellisGenerator] TrellisImageTo3DPipeline loaded in {time.time() - t0:.2f}s")
                return
            except ImportError:
                pass

            # Fallback: Load via HuggingFace transformers / diffusers or custom pipeline runner
            from huggingface_hub import snapshot_download
            repo_dir = snapshot_download(repo_id=self.model_id)
            print(f"[TrellisGenerator] Snapshot downloaded to {repo_dir}")
            
            # Use diffusers/transformers or custom loader
            self._is_loaded = True

        except Exception as e:
            print(f"[TrellisGenerator] Warning: Could not initialize native TRELLIS pipeline: {e}")
            self.pipeline = None
            self._is_loaded = False

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        """Loads and pre-processes input image to square RGBA/RGB format with centered subject."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        else:
            img = image_input

        img = img.convert("RGBA")
        
        # Crop transparent borders if present
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Pad to square aspect ratio
        w, h = img.size
        max_dim = max(w, h)
        square_img = Image.new("RGBA", (max_dim, max_dim), (255, 255, 255, 0))
        square_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

        # Resize to standard input resolution (1024x1024)
        square_img = square_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        return square_img

    def generate_3d_mesh(
        self,
        image_input: Union[str, Path, Image.Image],
        output_glb_path: str,
        seed: int = 42,
        decimate_target: int = 300000,
        resolution: str = "1024",
        texture_size: int = 4096,
        tex_slat_steps: int = 30,
        sparse_structure_steps: int = 12,
        shape_slat_steps: int = 12,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Converts 2D image into 3D GLB mesh using TRELLIS.2-4B.
        """
        t0 = time.time()
        output_path = Path(output_glb_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def report(pct: int, msg: str, step_idx: int = 1, total_steps: int = 5):
            if progress_callback and callable(progress_callback):
                try:
                    progress_callback(pct, msg, step_idx, total_steps)
                except Exception:
                    pass

        # Step 1: Preprocess input image
        report(10, "Tiền xử lý ảnh 2D (Tách nền Alpha & căn giữa bbox)", 1, 5)
        # If image_input is PIL Image, save to temporary file
        if isinstance(image_input, Image.Image):
            temp_img_path = output_path.parent / "temp_input_image.png"
            image_input.save(str(temp_img_path))
            input_file_str = str(temp_img_path.resolve())
        else:
            input_file_str = str(Path(image_input).resolve())

        mesh_generated = False
        trellis_python = sys.executable
        for _candidate in ["/venv/main/bin/python", "/home/braitoli/miniconda/envs/trellis/bin/python"]:
            if Path(_candidate).exists():
                trellis_python = _candidate
                break
        infer_script = Path(__file__).parent / "trellis2_infer.py"

        # Step 2: Initialize TRELLIS.2-4B
        report(25, "Khởi tạo mạng nơ-ron Transformer TRELLIS.2-4B...", 2, 5)

        # Check if Persistent GPU Worker is active on port 7865
        worker_port = int(os.environ.get("TRELLIS_PORT", "7865"))
        # STATUE_RESIDENT_GENERATOR=trellis means the caller expects the resident worker
        # to exist. If it doesn't answer, that's a real failure to surface, not a cue to
        # silently spin up a second 4B model in the same memory pool.
        resident_mode = os.environ.get("STATUE_RESIDENT_GENERATOR", "").strip().lower() == "trellis"
        is_worker_online = False
        try:
            import requests
            r_chk = requests.get(f"http://127.0.0.1:{worker_port}/health", timeout=1.0)
            if r_chk.status_code == 200 and r_chk.json().get("status") == "online":
                is_worker_online = True
        except Exception:
            is_worker_online = False

        if is_worker_online:
            try:
                print(f"[TrellisGenerator] ⚡ FAST PATH: Invoking Warm GPU Worker (port {worker_port})...")
                import threading
                call_state = {"done": False, "res": None, "err": None}

                def call_worker():
                    try:
                        import requests
                        payload = {
                            "image_path": str(input_file_str),
                            "output_path": str(output_path),
                            "seed": seed,
                            "resolution": resolution,
                            "decimation_target": min(decimate_target, 300000) if decimate_target else 300000,
                            "texture_size": texture_size,
                            "tex_slat_steps": tex_slat_steps,
                            "sparse_structure_steps": sparse_structure_steps,
                            "shape_slat_steps": shape_slat_steps
                        }
                        # Must exceed worst-case bake time (texture_size=4096 measured at ~360s,
                        # slower when the GPU is shared); a premature timeout drops us onto the
                        # subprocess path, which reloads the whole 4B model from scratch.
                        resp = requests.post(f"http://127.0.0.1:{worker_port}/generate", json=payload, timeout=1800)

                        if resp.status_code == 200:
                            call_state["res"] = resp.json()
                        else:
                            call_state["err"] = resp.text
                    except Exception as e:
                        call_state["err"] = str(e)
                    finally:
                        call_state["done"] = True

                w_thread = threading.Thread(target=call_worker, daemon=True)
                w_thread.start()

                t_w_start = time.time()
                # Two terms, because the two detail knobs pay for different things: the
                # flow-matching pass scales with the grid, the PBR bake with texture area.
                # A single 360s constant was calibrated for 1024/4096, and left a preview
                # run sitting at 25% for its whole life.
                est_duration = (120.0 if str(resolution) == "512" else 240.0) \
                    + 360.0 * (texture_size / 4096.0) ** 2
                while not call_state["done"]:
                    elapsed = time.time() - t_w_start
                    ratio = min(1.0, elapsed / est_duration)
                    cur_pct = int(25 + 50 * (1.0 - np.exp(-2.2 * ratio)))
                    cur_pct = min(75, max(25, cur_pct))
                    report(cur_pct, "Đang chạy Flow-Matching Latent Slats Diffusion 4B (1024 Cascade)...", 3, 5)
                    time.sleep(0.3)

                w_thread.join()
                if call_state["res"] and output_path.exists() and output_path.stat().st_size > 1000:
                    report(85, "Sinh lưới bề mặt PBR qua O-Voxel và khử đa giác thừa...", 4, 5)
                    print(f"[TrellisGenerator] ⚡ GPU Worker finished in {call_state['res'].get('total_time_sec', 0)}s!")
                    mesh_generated = True
                else:
                    print(f"[TrellisGenerator] Worker returned error: {call_state.get('err')}, falling back to CLI subprocess...")
            except Exception as e_w:
                print(f"[TrellisGenerator] Error calling GPU worker: {e_w}")

        if not mesh_generated and resident_mode:
            raise RuntimeError(
                f"Chế độ resident TRELLIS đang bật (STATUE_RESIDENT_GENERATOR=trellis) nhưng "
                f"worker cổng {worker_port} không sẵn sàng. Không tự động tải thêm một bản "
                f"TRELLIS.2-4B dự phòng để tránh hai bản model cùng tồn tại trong bộ nhớ — "
                f"thử lại sau khi worker online, hoặc tắt STATUE_RESIDENT_GENERATOR."
            )

        if not mesh_generated and Path(trellis_python).exists() and infer_script.exists():
            try:
                print(f"[TrellisGenerator] Invoking TRELLIS.2-4B neural engine via subprocess on {input_file_str}...")
                import subprocess
                import threading

                cmd = [
                    trellis_python,
                    str(infer_script),
                    f"--image_path={input_file_str}",
                    f"--output_path={output_path}",
                    f"--seed={seed}",
                    f"--resolution={resolution}",
                    f"--decimation_target={min(decimate_target, 300000) if decimate_target else 300000}",
                    # Passed explicitly now: leaving it off meant this fallback path baked at
                    # trellis2_infer's own 4096 default whatever the worker path was asked for.
                    f"--texture_size={texture_size}",
                    f"--tex_slat_steps={tex_slat_steps}",
                    f"--sparse_structure_steps={sparse_structure_steps}",
                    f"--shape_slat_steps={shape_slat_steps}",
                ]
                env = os.environ.copy()
                env["PATH"] = f"/home/braitoli/miniconda/envs/trellis/bin:{env.get('PATH', '')}"
                env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                env["ATTN_BACKEND"] = os.environ.get("ATTN_BACKEND", "sdpa")
                try:
                    import xformers
                    _default_sparse = "xformers"
                except ImportError:
                    _default_sparse = "sdpa"
                env["SPARSE_ATTN_BACKEND"] = os.environ.get("SPARSE_ATTN_BACKEND", _default_sparse)

                proc_state = {"done": False, "res": None}

                def run_sub():
                    proc_state["res"] = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    proc_state["done"] = True

                sub_t = threading.Thread(target=run_sub, daemon=True)
                sub_t.start()

                # Smoothly update progress from 25% to 75% while subprocess is calculating
                t_sub_start = time.time()
                est_duration = 18.0
                while not proc_state["done"]:
                    elapsed = time.time() - t_sub_start
                    ratio = min(1.0, elapsed / est_duration)
                    cur_pct = int(25 + 50 * (1.0 - np.exp(-2.2 * ratio)))
                    cur_pct = min(75, max(25, cur_pct))
                    report(cur_pct, "Đang chạy Flow-Matching Latent Slats Diffusion 4B...", 3, 5)
                    time.sleep(0.4)

                sub_t.join()
                res = proc_state["res"]
                if res and res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000:
                    report(85, "Sinh lưới bề mặt PBR qua O-Voxel và khử đa giác thừa...", 4, 5)
                    print(f"[TrellisGenerator] TRELLIS.2-4B successfully generated 3D model ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
                    mesh_generated = True
                else:
                    print(f"[TrellisGenerator] TRELLIS.2 subprocess returned {getattr(res, 'returncode', -1)}:\n{getattr(res, 'stderr', '')}")
            except Exception as e:
                print(f"[TrellisGenerator] Error executing TRELLIS.2 subprocess: {e}")

        # Không dùng mesh dựng hình học giả (SDF từ silhouette 2D) thay thế nữa: cả worker
        # resident lẫn subprocess tự tải đều thất bại nghĩa là TRELLIS thật sự không sinh
        # được mesh — báo lỗi rõ ràng để người dùng biết job thất bại, thay vì âm thầm trả
        # về một kết quả không phải TRELLIS mà trông giống thành công.
        if not mesh_generated or not output_path.exists():
            raise RuntimeError(
                "TRELLIS.2-4B không sinh được mesh (worker resident và subprocess tự tải đều "
                "thất bại)."
            )

        # Step 5: Inspect created mesh statistics & auto-orient to upright Y-Up
        report(95, "Căn chỉnh hệ toạ độ Y-Up & chiếu xạ kết cấu màu sắc (UV/Texture)...", 5, 5)
        scene_or_mesh = trimesh.load(str(output_path), force="mesh", process=False)
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh_final = scene_or_mesh.dump(concatenate=True)
        else:
            mesh_final = scene_or_mesh

        # If model is lying horizontally (Z-extent is height instead of Y), rotate upright
        ext = mesh_final.extents
        if ext[2] > ext[1] * 1.15:
            rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
            mesh_final.apply_transform(rot_x)

        # TRELLIS bakes a full PBR atlas (baseColor + metallicRoughness) via o_voxel.
        # Only fall back to flat 2D projection when that atlas is absent (fallback mesh),
        # since the projection ignores depth and would paint the back with front colors.
        has_baked_texture = (
            isinstance(mesh_final.visual, trimesh.visual.TextureVisuals)
            and mesh_final.visual.uv is not None
            and getattr(mesh_final.visual.material, "baseColorTexture", None) is not None
        )
        if has_baked_texture:
            print(f"[TrellisGenerator] Keeping baked PBR atlas "
                  f"{mesh_final.visual.material.baseColorTexture.size} from TRELLIS.2")
            mesh_final.export(str(output_path), extension_webp=True)
        else:
            if isinstance(image_input, Image.Image):
                pil_img = image_input
            else:
                pil_img = Image.open(input_file_str)
            mesh_final = self._project_image_texture(mesh_final, pil_img)
            mesh_final.export(str(output_path))

        t1 = time.time()
        report(100, "Đã hoàn thành tạo mô hình 3D kèm Texture màu sắc!", 5, 5)

        return {
            "output_glb_path": str(output_path),
            "num_vertices": len(mesh_final.vertices),
            "num_faces": len(mesh_final.faces),
            "generation_time_sec": round(t1 - t0, 2),
            "model_used": self.model_id
        }

    def _project_image_texture(self, mesh: trimesh.Trimesh, img: Image.Image) -> trimesh.Trimesh:
        """
        Projects high-resolution 2D character image onto 3D mesh vertices for rich full-color texture rendering.
        """
        try:
            w, h = img.size
            orig_arr = np.array(img.convert("RGBA"))
            verts = np.array(mesh.vertices)
            
            min_xy = verts[:, :2].min(axis=0)
            max_xy = verts[:, :2].max(axis=0)
            span_xy = np.maximum(max_xy - min_xy, 1e-5)
            
            norm_x = (verts[:, 0] - min_xy[0]) / span_xy[0]
            norm_y = (verts[:, 1] - min_xy[1]) / span_xy[1]
            
            norm_x = 0.05 + 0.90 * np.clip(norm_x, 0.0, 1.0)
            norm_y = 0.05 + 0.90 * np.clip(norm_y, 0.0, 1.0)
            
            uv_x = np.clip((norm_x * (w - 1)).astype(int), 0, w - 1)
            uv_y = np.clip(((1.0 - norm_y) * (h - 1)).astype(int), 0, h - 1)
            
            colors = orig_arr[uv_y, uv_x, :]
            colors[:, 3] = 255
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        except Exception as e:
            print(f"[TrellisGenerator] Texture projection note: {e}")
        return mesh
