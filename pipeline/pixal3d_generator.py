import os
import sys
import time
import signal
import subprocess
import threading
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Union, Dict, Any, Callable, List

# This host is a GB10: the GPU allocates out of the same LPDDR5X the OS runs on, so
# "free VRAM" is just free system memory and a resident model server is holding pages
# the GPU cannot get back. Measured here while chasing a CUDA OOM at
# Pixal3D/pixal3d/pipelines/base.py:69, on a 121.63 GB pool:
#
#   MemFree 12.4 GB -> torch.cuda.mem_get_info free 11.66 GB   (TRELLIS worker resident)
#   MemFree 37.4 GB -> torch.cuda.mem_get_info free 37.11 GB   (worker stopped)
#   MemFree 75.2 GB -> torch.cuda.mem_get_info free 72.30 GB   (page cache dropped)
#
# so MemFree tracks what a .cuda() call can actually get, to within ~0.5 GB, while
# MemAvailable does not: a reading of 28 GB available / 11.66 GB free still OOMed.
#
# Standard mode costs far more than the ~34 GB the load itself accounts for
# (from_pretrained builds the four DiT stages on the CPU, ~17 GB resident, and
# pipeline.cuda() then holds the device copy alongside it), because the kernel also has
# to find pages for the 21 GB of weights being read through the page cache while swap is
# already full. Observed: OOM at 37.4 GB free, OOM again at ~49 GB, success at 75.2 GB.
# The threshold sits above the second failure and below the only success rather than at
# the load's nominal cost -- and it is deliberately pessimistic, because guessing wrong
# in this direction only picks low-VRAM mode, which measured the same speed and the same
# mesh (see below), while guessing wrong the other way costs a ~90s doomed load.
PIXAL3D_STANDARD_FREE_GB = 70.0

# Low-VRAM mode never materialises that second full copy -- each stage is moved to the
# device for its own sampling pass and back to the CPU afterwards (see the low_vram
# branches in pixal3d/pipelines/pixal3d_image_to_3d.py) -- so it needs the resident copy
# plus the largest single stage rather than twice everything. Same weights and sampler
# steps, and on this host the host-device copies are nearly free because there is only
# one pool: the same image at 1536 took 400.2s in low-VRAM against 403.9s standard, 993k
# faces against 971k, same 4096 bake. It is a memory trade, not a quality one.
#
# It is not a small appetite though, and this number is only for the error message. A
# low-VRAM run starting from 49.3 GB free finished (dipping to 1.4 GB mid-run), while one
# starting from ~45 GB died at image_conditioned_proj.py:523 moving a DinoV3 encoder onto
# the device -- the page cache left behind by the failed standard attempt is the
# difference, and MemFree at the start does not capture it.
PIXAL3D_LOW_VRAM_FREE_GB = 48.0

# playground/server.py starts this at boot to keep TRELLIS.2-4B warm, and it holds
# ~25 GB of the pool for as long as the server lives.
_TRELLIS_WORKER_SCRIPT = "trellis_worker_service.py"
_PIXAL3D_SCRIPTS = ("pixal3d_infer.py", "pixal3d_mv_infer.py")

# Deliberately not torch.AcceleratorError, which is every CUDA error: mistaking an
# illegal access for an OOM would both mislead the user and buy a pointless retry.
_OOM_MARKERS = (
    "CUDA error: out of memory",
    "CUDA out of memory",
    "cudaErrorMemoryAllocation",
)


def _mem_free_gb() -> float:
    """MemFree, in GiB. See PIXAL3D_STANDARD_FREE_GB for why it is MemFree and not
    MemAvailable."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemFree:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("inf")


def _pids_running(scripts) -> List[int]:
    """PIDs of python processes running any of these scripts."""
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode("utf-8", "ignore").split("\0")
        except OSError:
            continue  # the process exited between the listdir and the read
        # argv[0] has to be the interpreter and the script has to be an argument, so a
        # shell that merely mentions the path in its command line is not a match.
        if len(argv) >= 2 and "python" in argv[0] and any(
            a.endswith(script) for a in argv[1:] for script in scripts
        ):
            pids.append(int(entry.name))
    return pids


def _trellis_worker_pids() -> List[int]:
    return _pids_running((_TRELLIS_WORKER_SCRIPT,))


def pixal3d_is_running() -> bool:
    """True while a Pixal3D inference subprocess is alive.

    playground/server.py checks this before bringing the TRELLIS worker back: jobs run
    concurrently in the background thread pool, and putting ~25 GB back into the pool
    underneath a Pixal3D run that is still sampling out of it would cause exactly the
    OOM this module exists to prevent.
    """
    return bool(_pids_running(_PIXAL3D_SCRIPTS))


def stop_trellis_worker() -> float:
    """Stop the persistent TRELLIS worker and return the GiB of MemFree it gave back.

    The worker is a cache, not a dependency: trellis_generator.py polls its /health and
    falls back to a subprocess that loads TRELLIS.2-4B itself when the worker is offline,
    measured at +90.7s. So this costs the next TRELLIS job a model load rather than
    costing this job everything.

    Also used by playground/server.py to evict the resident model when a job for a
    different generator arrives -- see apply_resident_policy() there. It only ever
    matches a python process running trellis_worker_service.py, never the shared TRELLIS
    service on port 7870 that belongs to another project.
    """
    pids = _trellis_worker_pids()
    if not pids:
        return 0.0
    before = _mem_free_gb()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + 25.0
    while time.time() < deadline and _trellis_worker_pids():
        time.sleep(0.5)
    for pid in _trellis_worker_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(3.0)  # the pages come back a moment after the process does
    return _mem_free_gb() - before


def _model_blob_dir() -> Optional[Path]:
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub:
        root = Path(hub)
    elif os.environ.get("HF_HOME"):
        root = Path(os.environ["HF_HOME"]) / "hub"
    else:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    blobs = root / "models--TencentARC--Pixal3D" / "blobs"
    return blobs if blobs.is_dir() else None


def _drop_model_page_cache() -> float:
    """Ask the kernel to forget the cached pages of the Pixal3D weights, returning the
    GiB of MemFree that came back.

    The CUDA driver will not reclaim page cache to satisfy an allocation -- 28 GB of
    MemAvailable sat beside 11.66 GB of MemFree while .cuda() failed -- so the ~40 GB of
    weights a previous run pulled through the cache stays in the way, and an OOMed
    attempt leaves the retry worse off than the first try. Unlike drop_caches this needs
    no privileges and touches only these files, which the run is about to read anyway,
    so the cost is one cold read. Measured on this host at +15.8 GB in 3.5s.
    """
    blobs = _model_blob_dir()
    if blobs is None:
        return 0.0
    before = _mem_free_gb()
    for path in blobs.iterdir():
        if not path.is_file():
            continue
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        finally:
            os.close(fd)
    time.sleep(1.0)  # the counters settle a moment behind the eviction
    return _mem_free_gb() - before


def _plan_memory(requested_low_vram: bool, report=None) -> bool:
    """Make room for the load and decide which mode it can afford.

    Returns the low-VRAM flag actually to use, which may be True even when the caller
    asked for standard mode.

    There are only two levers here, and both get pulled before the mode is chosen: the
    cached model pages, and the warm TRELLIS worker. Everything else is out of reach --
    the 25 GB TRELLIS service on port 7870 belongs to another project and swap is
    already full. The worker goes whenever the standard threshold is not met rather than
    only when low-VRAM also fails to fit, because a run that kept the cache warm and then
    OOMed halfway through sampling -- observed, at image_conditioned_proj.py:523 -- traded
    the user's job for a cache.
    """
    free = _mem_free_gb()

    if free < PIXAL3D_STANDARD_FREE_GB:
        recovered = _drop_model_page_cache()
        if recovered > 0.5:
            free = _mem_free_gb()
            print(f"[Pixal3D] Dropped the cached model pages: {recovered:.1f} GB back, "
                  f"now {free:.1f} GB free.")

    if free < PIXAL3D_STANDARD_FREE_GB and _trellis_worker_pids():
        if report:
            # Same percentage as the caller's last report: this is a pause inside that
            # step, and a bar that walks backwards reads as a restart.
            report(25, "Đang giải phóng bộ nhớ GPU (tạm dừng worker TRELLIS)...", 2, 5)
        freed = stop_trellis_worker()
        free = _mem_free_gb()
        print(f"[Pixal3D] Stopped the persistent TRELLIS worker for headroom: it gave "
              f"back {freed:.1f} GB, now {free:.1f} GB free.")

    if requested_low_vram or free < PIXAL3D_STANDARD_FREE_GB:
        if not requested_low_vram:
            print(f"[Pixal3D] {free:.1f} GB free is below the "
                  f"{PIXAL3D_STANDARD_FREE_GB:.0f} GB a standard-mode load needs; using "
                  f"low-VRAM mode (same weights and sampler steps, measured at the same "
                  f"wall time and the same mesh on this host).")
        return True
    return False


def _oom_message(err_msg: str, was_low_vram: bool) -> Optional[str]:
    """A message the studio can show as-is, or None when this was not an OOM."""
    if not any(marker in err_msg for marker in _OOM_MARKERS):
        return None
    mode = "chế độ tiết kiệm bộ nhớ" if was_low_vram else "chế độ tiêu chuẩn"
    needed = PIXAL3D_LOW_VRAM_FREE_GB if was_low_vram else PIXAL3D_STANDARD_FREE_GB
    return (
        f"GPU không đủ bộ nhớ để chạy Pixal3D ({mode}): chỉ còn {_mem_free_gb():.1f} GB "
        f"trống, cần khoảng {needed:.0f} GB. Máy này dùng bộ nhớ hợp nhất nên GPU phải "
        f"chia sẻ RAM với mọi tiến trình khác. Hãy đóng bớt các tiến trình AI đang chạy "
        f"rồi thử lại, hoặc chọn model TRELLIS cho tượng này. Lưu ý: hạ 'Chất lượng "
        f"Mesh' xuống 'Nhanh' KHÔNG giúp được ở đây — mọi mức đều nạp cùng bộ trọng số."
    )


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

        low_vram = _plan_memory(low_vram, report)

        def build_cmd(use_low_vram: bool):
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
            if use_low_vram:
                cmd.append("--low_vram")
            return cmd

        cmd = build_cmd(low_vram)

        env = os.environ.copy()
        env["PATH"] = f"/home/braitoli/miniconda/envs/trellis/bin:{env.get('PATH', '')}"
        env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env["ATTN_BACKEND"] = "flash_attn"

        # Measured on a GB10 at 1536 / 12 steps per stage / 1M-face decimation / 4096
        # bake: 403.9s standard and 400.2s low-VRAM, so one estimate covers both modes.
        # The retry below reuses this, hence a closure rather than an inline loop -- a
        # retry that reported nothing would leave the bar parked for a whole second run.
        est_duration = 450.0 if str(resolution) == "1536" else 150.0

        def run_with_progress(run_cmd, message):
            proc_state = {"done": False, "res": None}

            def run_sub():
                proc_state["res"] = subprocess.run(run_cmd, capture_output=True, text=True, env=env)
                proc_state["done"] = True

            sub_t = threading.Thread(target=run_sub, daemon=True)
            sub_t.start()

            # Smoothly update progress from 25% to 75%
            t_sub_start = time.time()
            while not proc_state["done"]:
                ratio = min(1.0, (time.time() - t_sub_start) / est_duration)
                cur_pct = int(25 + 50 * (1.0 - np.exp(-2.2 * ratio)))
                report(min(75, max(25, cur_pct)), message, 3, 5)
                time.sleep(0.4)

            sub_t.join()
            return proc_state["res"]

        res = run_with_progress(cmd, "Đang chạy Pixal3D Pixel-Aligned Transformer Diffusion...")

        def succeeded(r):
            return bool(r) and r.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000

        if not succeeded(res):
            err_msg = (res.stderr or res.stdout or "Unknown error") if res else "Unknown error"
            # Another process can take the memory between _plan_memory's reading and the
            # load, so an OOM here is not necessarily a wrong plan -- it is a race, and
            # low-VRAM mode is the same result for more wall time rather than a lesser one.
            if _oom_message(err_msg, low_vram) and not low_vram:
                print(f"[Pixal3D] Standard-mode load ran out of memory with "
                      f"{_mem_free_gb():.1f} GB free; retrying in low-VRAM mode.")
                # The attempt that just died pulled ~21 GB of weights through the page
                # cache on its way out, which is what made the first retry here fail
                # worse than the original try. Hand those pages back before going again.
                recovered = _drop_model_page_cache()
                print(f"[Pixal3D] Dropped the cached model pages before the retry: "
                      f"{recovered:.1f} GB back, now {_mem_free_gb():.1f} GB free.")
                low_vram = True
                res = run_with_progress(
                    build_cmd(True),
                    "Thiếu bộ nhớ GPU — chạy lại ở chế độ tiết kiệm bộ nhớ...",
                )
                err_msg = (res.stderr or res.stdout or "Unknown error") if res else "Unknown error"

        if succeeded(res):
            report(85, "Sinh lưới bề mặt PBR qua O-Voxel và khử đa giác thừa...", 4, 5)
            print(f"[Pixal3D] Model 3D tạo thành công: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
            mesh_generated = True
        else:
            print(f"[Pixal3D] Error during inference:\n{err_msg}")
            friendly = _oom_message(err_msg, low_vram)
            if friendly:
                raise RuntimeError(friendly)
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

        low_vram = _plan_memory(low_vram, report)

        def build_cmd(use_low_vram: bool):
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
            if use_low_vram:
                cmd.append("--low_vram")
            return cmd

        cmd = build_cmd(low_vram)

        env = os.environ.copy()
        env["PATH"] = f"/home/braitoli/miniconda/envs/trellis/bin:{env.get('PATH', '')}"
        env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env["ATTN_BACKEND"] = "flash_attn"

        # Measured on a GB10: 457s for the 1536 cascade at 12 steps/stage with a 1M-face
        # decimation and a 4096 bake. A 50s estimate here pinned the bar at 80% for most
        # of the run, which reads as a hang.
        est_duration = 450.0 if str(resolution) == "1536" else 150.0

        def run_with_progress(run_cmd, message):
            proc_state = {"done": False, "res": None}

            def run_sub():
                proc_state["res"] = subprocess.run(run_cmd, capture_output=True, text=True, env=env)
                proc_state["done"] = True

            sub_t = threading.Thread(target=run_sub, daemon=True)
            sub_t.start()

            t_sub_start = time.time()
            while not proc_state["done"]:
                ratio = min(1.0, (time.time() - t_sub_start) / est_duration)
                cur_pct = int(25 + 55 * (1.0 - np.exp(-2.0 * ratio)))
                report(min(80, max(25, cur_pct)), message, 3, 5)
                time.sleep(0.4)

            sub_t.join()
            return proc_state["res"]

        res = run_with_progress(cmd, "Đang chạy Pixal3D Multi-View Flow Matching 360°...")

        def succeeded(r):
            return bool(r) and r.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1000

        if not succeeded(res):
            err_msg = (res.stderr or res.stdout or "Unknown error") if res else "Unknown error"
            if _oom_message(err_msg, low_vram) and not low_vram:
                print(f"[Pixal3D-MV] Standard-mode load ran out of memory with "
                      f"{_mem_free_gb():.1f} GB free; retrying in low-VRAM mode.")
                # The attempt that just died pulled ~21 GB of weights through the page
                # cache on its way out, which is what made the first retry here fail
                # worse than the original try. Hand those pages back before going again.
                recovered = _drop_model_page_cache()
                print(f"[Pixal3D] Dropped the cached model pages before the retry: "
                      f"{recovered:.1f} GB back, now {_mem_free_gb():.1f} GB free.")
                low_vram = True
                res = run_with_progress(
                    build_cmd(True),
                    "Thiếu bộ nhớ GPU — chạy lại ở chế độ tiết kiệm bộ nhớ...",
                )
                err_msg = (res.stderr or res.stdout or "Unknown error") if res else "Unknown error"

        if succeeded(res):
            report(90, "Hoàn tất tái tạo khối 3D 360° từ 4 góc nhìn!", 4, 5)
            print(f"[Pixal3D-MV] Model 3D Multi-view tạo thành công: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"[Pixal3D-MV] Error during MV inference:\n{err_msg}")
            friendly = _oom_message(err_msg, low_vram)
            if friendly:
                raise RuntimeError(friendly)
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

