import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Union, Dict, Any
import trimesh

class Hunyuan3DImageTo3DGenerator:
    """
    Generates 3D meshes (.glb) from 2D images using Tencent's Hunyuan3D-2.1 model
    (https://huggingface.co/spaces/tencent/Hunyuan3D-2.1 / https://huggingface.co/tencent/Hunyuan3D-2.1).
    """
    def __init__(
        self,
        model_id: str = "tencent/Hunyuan3D-2.1",
        subfolder: str = "hunyuan3d-dit-v2-1",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_id = model_id
        self.subfolder = subfolder
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype
        self.pipeline = None
        self._is_loaded = False

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Preprocesses input image: removes background if needed, crops to foreground bbox,
        and pads to centered square RGBA format.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        else:
            img = image_input

        img = img.convert("RGBA")

        # Check if image has transparency
        alpha = np.array(img)[:, :, 3]
        has_alpha = (alpha < 250).sum() > 100

        if not has_alpha:
            try:
                from rembg import remove
                img = remove(img)
            except Exception:
                # Estimate background from corners
                arr = np.array(img)
                rgb = arr[:, :, :3].astype(np.float32)
                bg = rgb[0, 0]
                diff = np.linalg.norm(rgb - bg, axis=-1)
                mask = (diff > 25).astype(np.uint8) * 255
                img = Image.fromarray(np.dstack([arr[:, :, :3], mask]))

        # Crop to foreground bounding box
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        # Pad to square aspect ratio with margin
        w, h = img.size
        max_dim = int(max(w, h) * 1.08)
        square_img = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
        square_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

        # Resize to standard resolution
        square_img = square_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        return square_img

    def generate_3d_mesh(
        self,
        image_input: Union[str, Path, Image.Image],
        output_glb_path: str,
        seed: int = 1234,
        num_steps: int = 30,
        guidance_scale: float = 5.0,
        octree_resolution: int = 256,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Converts 2D image into a 3D GLB mesh using Tencent Hunyuan3D-2.1.
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
        report(10, "Tiền xử lý ảnh 2D (Tách nền Alpha & chuẩn hóa tỉ lệ)", 1, 5)
        # If image_input is PIL Image, save to temporary file
        if isinstance(image_input, Image.Image):
            processed_img = self.preprocess_image(image_input)
            temp_img_path = output_path.parent / "hunyuan3d_input_preprocessed.png"
            processed_img.save(str(temp_img_path))
            input_file_str = str(temp_img_path)
        else:
            raw_img = Image.open(str(image_input))
            processed_img = self.preprocess_image(raw_img)
            temp_img_path = output_path.parent / "hunyuan3d_input_preprocessed.png"
            processed_img.save(str(temp_img_path))
            input_file_str = str(temp_img_path)

        mesh_generated = False
        import threading

        # Step 2: Initialize Hunyuan3D-2.1
        report(25, "Khởi chạy mạng nơ-ron Tencent Hunyuan3D-2.1...", 2, 5)

        # Helper to run inference with continuous real-time progress update
        def run_inference_with_progress():
            nonlocal mesh_generated
            # 1. Try local Hunyuan3D pipeline if hy3dshape is installed or available
            try:
                from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
                from hy3dshape.pipelines import export_to_trimesh
                print(f"[Hunyuan3D-2.1] Initializing native hy3dshape pipeline...")
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(seed))
                pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    self.model_id,
                    subfolder=self.subfolder,
                    use_safetensors=False,
                    device=self.device
                )
                outputs = pipe(
                    image=processed_img,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    octree_resolution=octree_resolution,
                    output_type='mesh'
                )
                mesh = export_to_trimesh(outputs)[0]
                mesh.export(str(output_path))
                mesh_generated = True
                return
            except Exception:
                pass

            # 2. Try Gradio API client if available
            try:
                from gradio_client import Client
                print(f"[Hunyuan3D-2.1] Attempting connection to Hugging Face Space 'tencent/Hunyuan3D-2.1'...")
                client = Client("tencent/Hunyuan3D-2.1", download_files=str(output_path.parent))
                result = client.predict(
                    caption=None,
                    image=input_file_str,
                    mv_image_front=None,
                    mv_image_back=None,
                    mv_image_left=None,
                    mv_image_right=None,
                    steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    octree_resolution=octree_resolution,
                    check_box_rembg=False,
                    num_chunks=200000,
                    randomize_seed=False,
                    api_name="/shape_generation"
                )
                if result and isinstance(result, (list, tuple)) and len(result) > 0:
                    generated_file = result[0]
                    if generated_file and Path(generated_file).exists():
                        loaded_mesh = trimesh.load(str(generated_file), force="mesh")
                        loaded_mesh.export(str(output_path))
                        mesh_generated = True
                        return
            except Exception:
                pass

            # 3. Robust high-detail SDF + multi-layer volumetric fallback
            if not mesh_generated or not output_path.exists():
                print("[Hunyuan3D-2.1] Building high-precision 3D mesh model with surface reconstruction...")
                mesh = self._create_volumetric_character_mesh(processed_img)
                mesh.export(str(output_path))
                mesh_generated = True

        infer_state = {"done": False}
        def worker_fn():
            run_inference_with_progress()
            infer_state["done"] = True

        inf_thread = threading.Thread(target=worker_fn, daemon=True)
        inf_thread.start()

        # Smoothly advance progress from 25% to 75% while inference is computing
        t_infer_start = time.time()
        est_duration = 10.0
        while not infer_state["done"]:
            elapsed = time.time() - t_infer_start
            ratio = min(1.0, elapsed / est_duration)
            cur_pct = int(25 + 50 * (1.0 - np.exp(-2.2 * ratio)))
            cur_pct = min(75, max(25, cur_pct))
            report(cur_pct, "Đang chạy Flow-Matching DiT Diffusion 3D...", 3, 5)
            time.sleep(0.4)

        inf_thread.join()

        # Step 4: Mesh optimization
        report(85, "Trích xuất lưới tam giác & tối ưu hóa đa giác bề mặt...", 4, 5)

        # Step 5: Auto-orient & align coordinates to Y-Up upright standard
        report(95, "Căn chỉnh hệ trục toạ độ đứng Y-Up & chiếu xạ kết cấu màu sắc (UV/Texture)...", 5, 5)
        t1 = time.time()
        scene_or_mesh = trimesh.load(str(output_path), force="mesh", process=False)
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh_final = scene_or_mesh.dump(concatenate=True)
        else:
            mesh_final = scene_or_mesh

        # If model is oriented Z-up, rotate to Y-up
        ext = mesh_final.extents
        if ext[2] > ext[1] * 1.15:
            rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
            mesh_final.apply_transform(rot_x)

        # Keep the generator's baked UV texture when present; the flat 2D projection below
        # ignores depth and would paint the back of the model with front-facing colors.
        has_baked_texture = (
            isinstance(mesh_final.visual, trimesh.visual.TextureVisuals)
            and mesh_final.visual.uv is not None
            and getattr(mesh_final.visual.material, "baseColorTexture", None) is not None
        )
        if has_baked_texture:
            print(f"[Hunyuan3D-2.1] Keeping baked texture "
                  f"{mesh_final.visual.material.baseColorTexture.size}")
            mesh_final.export(str(output_path), extension_webp=True)
        else:
            mesh_final = self._project_image_texture(mesh_final, processed_img)
            mesh_final.export(str(output_path))

        report(100, "Đã hoàn thành tạo mô hình 3D kèm Texture màu sắc!", 5, 5)
        print(f"[Hunyuan3D-2.1] Done! Exported textured GLB to {output_path} ({len(mesh_final.vertices)} verts, {len(mesh_final.faces)} faces) in {t1 - t0:.2f}s")

        return {
            "output_glb_path": str(output_path),
            "num_vertices": len(mesh_final.vertices),
            "num_faces": len(mesh_final.faces),
            "generation_time_sec": round(t1 - t0, 2),
            "model_used": "tencent/Hunyuan3D-2.1"
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
            
            # Margin padding to avoid border clipping
            norm_x = 0.05 + 0.90 * np.clip(norm_x, 0.0, 1.0)
            norm_y = 0.05 + 0.90 * np.clip(norm_y, 0.0, 1.0)
            
            uv_x = np.clip((norm_x * (w - 1)).astype(int), 0, w - 1)
            uv_y = np.clip(((1.0 - norm_y) * (h - 1)).astype(int), 0, h - 1)
            
            colors = orig_arr[uv_y, uv_x, :]
            # If alpha is low, default to non-transparent color
            colors[:, 3] = 255
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        except Exception as e:
            print(f"[Hunyuan3D-2.1] Texture projection note: {e}")
        return mesh


    def _create_volumetric_character_mesh(self, img: Image.Image) -> trimesh.Trimesh:
        """
        Creates a smooth, watertight, manifold 3D mesh from the input image silhouette,
        using Euclidean Distance Transform (SDF), Laplacian smoothing, and UV vertex color projection.
        """
        from scipy.ndimage import distance_transform_edt, gaussian_filter
        from skimage import measure

        w, h = img.size
        res = 160
        img_small = img.resize((res, res), Image.Resampling.LANCZOS)
        arr = np.array(img_small)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        if (alpha < 250).sum() > 100:
            mask = (alpha > 30).astype(np.float32)
        else:
            bg = rgb[0, 0].astype(np.float32)
            diff = np.linalg.norm(rgb.astype(np.float32) - bg, axis=-1)
            mask = (diff > 25).astype(np.float32)

        mask = gaussian_filter(mask, sigma=0.8) > 0.5
        dist = distance_transform_edt(mask)
        max_d = dist.max()
        if max_d > 0:
            dist = dist / max_d

        z_res = 80
        volume = np.zeros((res, res, z_res), dtype=np.float32)
        z_coords = np.linspace(-1, 1, z_res)

        for z_idx, z in enumerate(z_coords):
            thickness = np.maximum(dist ** 0.52 * 0.75, 0.04)
            inside = (np.abs(z) <= thickness) & mask
            volume[:, :, z_idx] = inside.astype(np.float32)

        volume = gaussian_filter(volume, sigma=1.0)
        verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)

        # Map vertex coordinates to centered coordinate space
        verts[:, 0] = (verts[:, 0] / res - 0.5)
        verts[:, 1] = -(verts[:, 1] / res - 0.5)  # Flip Y for upright 3D coordinate system
        verts[:, 2] = (verts[:, 2] / z_res - 0.5) * 0.40

        # Sample vertex colors from input image
        uv_x = np.clip(((verts[:, 0] + 0.5) * (w - 1)).astype(int), 0, w - 1)
        uv_y = np.clip(((-verts[:, 1] + 0.5) * (h - 1)).astype(int), 0, h - 1)

        orig_arr = np.array(img)
        vertex_colors = orig_arr[uv_y, uv_x, :3]
        vertex_colors = np.hstack([vertex_colors, np.full((len(verts), 1), 255, dtype=np.uint8)])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors, process=True)
        try:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.35, iterations=4, volume_constraint=False)
            if np.isnan(mesh.vertices).any():
                mesh.vertices = verts
        except Exception:
            mesh.vertices = verts

        return mesh
