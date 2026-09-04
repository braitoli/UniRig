import os
import time
import numpy as np
import requests
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
    ):
        self.model_id = model_id

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

        # Step 2: Gọi gx10-model-serving — service TRELLIS dùng chung cho các dự án trên
        # gx10, luôn thường trú (không cần tự quản lý vòng đời model ở phía UniRig nữa).
        # Xem docs/superpowers/specs/2026-09-04-gx10-model-serving-design.md (repo
        # 3d-studio). Không có nhánh dự phòng nào khác: nếu service không phản hồi hoặc
        # job thất bại, raise lỗi rõ ràng — không tự tải một bản TRELLIS.2-4B thứ hai
        # trong tiến trình này (đó chính là điều gây OOM mà service này sinh ra để diệt).
        report(25, "Khởi tạo mạng nơ-ron Transformer TRELLIS.2-4B...", 2, 5)

        base_url = os.environ.get("MODEL_SERVING_URL", "http://127.0.0.1:7900").rstrip("/")

        try:
            with open(input_file_str, "rb") as f:
                files = {"image": (Path(input_file_str).name, f, "image/png")}
                data = {
                    "seed": str(seed),
                    "resolution": str(resolution),
                    "decimate_target": str(min(decimate_target, 300000) if decimate_target else 300000),
                    "texture_size": str(texture_size),
                    "tex_slat_steps": str(tex_slat_steps),
                    "sparse_structure_steps": str(sparse_structure_steps),
                    "shape_slat_steps": str(shape_slat_steps),
                }
                resp = requests.post(f"{base_url}/v1/generate", files=files, data=data, timeout=30)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Không kết nối được tới gx10-model-serving tại {base_url}: {e}. "
                "Service phải đang chạy trước khi submit job Statue Studio."
            )
        if resp.status_code != 202:
            raise RuntimeError(
                f"gx10-model-serving từ chối request (HTTP {resp.status_code}): {resp.text}"
            )
        job_id = resp.json()["job_id"]
        print(f"[TrellisGenerator] gx10-model-serving nhận job {job_id}, đang poll trạng thái...")

        # Hai thành phần vì hai tham số chi tiết trả giá khác nhau: diffusion pass scale
        # theo lưới (resolution), PBR bake scale theo diện tích texture (texture_size).
        est_duration = (120.0 if str(resolution) == "512" else 240.0) \
            + 360.0 * (texture_size / 4096.0) ** 2
        # gx10-model-serving là hàng đợi FIFO dùng chung với các dự án khác trên máy —
        # nếu job của dự án khác chiếm hàng đợi lâu, service treo ở "running", hoặc
        # response thiếu key "state", vòng poll không có trần thời gian sẽ chờ vô hạn và
        # giữ luôn thread trong pool. 1800s khớp với timeout HTTP của worker cũ trước khi
        # migrate sang gx10-model-serving (xem git show 99146ad:pipeline/trellis_generator.py).
        poll_deadline_sec = max(1800.0, est_duration * 3)
        t_poll_start = time.time()
        job_state = None
        job_error = None
        last_queue_position = None
        while True:
            elapsed = time.time() - t_poll_start
            if elapsed > poll_deadline_sec:
                raise RuntimeError(
                    f"gx10-model-serving job {job_id} vượt quá thời gian chờ tối đa "
                    f"({poll_deadline_sec:.0f}s, đã chờ {elapsed:.0f}s). "
                    f"queue_position cuối cùng thấy được: {last_queue_position!r}. "
                    "Job có thể đang kẹt trong hàng đợi FIFO dùng chung hoặc treo ở "
                    "trạng thái running."
                )
            try:
                status_resp = requests.get(f"{base_url}/v1/jobs/{job_id}", timeout=10)
                status_resp.raise_for_status()
                status = status_resp.json()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"Mất kết nối tới gx10-model-serving khi đang chờ job {job_id}: {e}"
                )
            last_queue_position = status.get("queue_position", last_queue_position)
            job_state = status.get("state")
            if job_state in ("succeeded", "failed"):
                job_error = status.get("error")
                break
            ratio = min(1.0, elapsed / est_duration)
            cur_pct = int(25 + 50 * (1.0 - np.exp(-2.2 * ratio)))
            cur_pct = min(75, max(25, cur_pct))
            report(cur_pct, "Đang chạy Flow-Matching Latent Slats Diffusion 4B (gx10-model-serving)...", 3, 5)
            time.sleep(2.0)

        if job_state != "succeeded":
            raise RuntimeError(
                f"gx10-model-serving job {job_id} thất bại: {job_error or 'không rõ lý do'}"
            )

        # Step 4: tải GLB kết quả về đúng đường dẫn output pipeline này mong đợi — service
        # không ghi vào output_dir do client chỉ định, client phải tự tải qua HTTP.
        report(85, "Sinh lưới bề mặt PBR qua O-Voxel và khử đa giác thừa...", 4, 5)
        result_resp = requests.get(f"{base_url}/v1/jobs/{job_id}/result", timeout=120)
        result_resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(result_resp.content)

        if not output_path.exists() or output_path.stat().st_size <= 1000:
            raise RuntimeError(
                f"gx10-model-serving báo job {job_id} succeeded nhưng file GLB tải về rỗng "
                f"hoặc quá nhỏ ({output_path})."
            )
        print(f"[TrellisGenerator] gx10-model-serving hoàn tất job {job_id} trong "
              f"{time.time() - t_poll_start:.1f}s -> {output_path.stat().st_size / 1024 / 1024:.2f} MB")

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
