"""
Background Automation & Webhook Dispatcher Service.
Monitors an input directory for new 2D image uploads, automatically processes them
through the 3D Statue Painting Pipeline, publishes optimized GLBs to the output folder,
and dispatches Webhook notifications to external online painting apps / APIs.
"""

import os
import sys
import time
import json
import shutil
import asyncio
import threading
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any

from playground import database
from pipeline.statue_pipeline import Statue3DPipeline

STORAGE_DIR = Path(__file__).resolve().parent / "storage"
ROOT_DIR = Path(__file__).resolve().parent.parent

class StatueAutomationService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, server_port: int = 7860):
        if self._initialized:
            return
        self.server_port = server_port
        self.statue_pipeline = Statue3DPipeline(root_dir=str(ROOT_DIR))
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._initialized = True

    def get_status(self) -> Dict[str, Any]:
        cfg = database.get_automation_config()
        return {
            "is_running": self.is_running,
            "config": cfg,
            "active_jobs_count": len(self.active_jobs),
            "active_jobs": list(self.active_jobs.keys())
        }

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        print("🤖 [Statue Automation] Service started. Watching input folder...")

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("🛑 [Statue Automation] Service stopped.")

    def trigger_scan_now(self) -> List[str]:
        """Manually trigger immediate folder scan and return found files."""
        cfg = database.get_automation_config()
        input_folder = Path(cfg.get("input_folder", str(STORAGE_DIR / "automation/input")))
        input_folder.mkdir(parents=True, exist_ok=True)
        
        found = self._scan_new_images(input_folder)
        for img_p in found:
            threading.Thread(target=self.process_file, args=(img_p, cfg), daemon=True).start()
        return [str(p.name) for p in found]

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                cfg = database.get_automation_config()
                if not cfg.get("enabled", False):
                    # Sleep if disabled in config
                    time.sleep(2.0)
                    continue

                input_folder = Path(cfg.get("input_folder", str(STORAGE_DIR / "automation/input")))
                input_folder.mkdir(parents=True, exist_ok=True)
                output_folder = Path(cfg.get("output_folder", str(STORAGE_DIR / "automation/output")))
                output_folder.mkdir(parents=True, exist_ok=True)

                poll_interval = float(cfg.get("poll_interval_sec", 5.0))
                new_files = self._scan_new_images(input_folder)

                for img_path in new_files:
                    if self._stop_event.is_set():
                        break
                    self.process_file(img_path, cfg)

                # Wait for poll interval
                self._stop_event.wait(timeout=poll_interval)
            except Exception as e:
                print(f"⚠️ [Statue Automation Worker] Exception in loop: {e}")
                time.sleep(5.0)

    def _scan_new_images(self, input_folder: Path) -> List[Path]:
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        found = []
        try:
            for item in sorted(input_folder.iterdir()):
                if item.is_file() and item.suffix.lower() in image_exts:
                    # Ignore hidden, temp or lock files
                    if item.name.startswith(".") or item.name.endswith(".tmp") or item.name.endswith(".part"):
                        continue
                    # Check if file size is stable (not currently uploading/writing)
                    try:
                        size1 = item.stat().st_size
                        time.sleep(0.3)
                        size2 = item.stat().st_size
                        if size1 == size2 and size1 > 0:
                            found.append(item)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Statue Automation] Error scanning {input_folder}: {e}")
        return found

    def process_file(self, img_path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the statue pipeline on a detected image file and dispatches webhook."""
        stem = img_path.stem
        job_id = f"auto_statue_{int(time.time())}_{stem}"
        
        with self._lock:
            if job_id in self.active_jobs:
                return {}
            self.active_jobs[job_id] = {"filename": img_path.name, "start_time": time.time()}

        try:
            # 1. Create target work directory and copy input file
            job_work_dir = STORAGE_DIR / f"statue_jobs/{job_id}"
            job_work_dir.mkdir(parents=True, exist_ok=True)
            target_input = job_work_dir / img_path.name
            shutil.copyfile(str(img_path), str(target_input))

            # Move original input file to processed subfolder to prevent duplicate runs
            processed_dir = img_path.parent / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            dest_processed = processed_dir / f"{int(time.time())}_{img_path.name}"
            shutil.move(str(img_path), str(dest_processed))

            # 2. Register job in DB
            metadata = {
                "progress": {"pct": 5, "step_name": "Đang khởi tạo job tự động...", "step_idx": 1, "total_steps": 5},
                "source_file": str(dest_processed)
            }
            job = database.create_statue_job(
                job_id=job_id,
                title=f"Auto: {img_path.name}",
                input_filename=img_path.name,
                input_file_path=str(target_input),
                generator_type=cfg.get("generator", "trellis"),
                mesh_detail=cfg.get("mesh_detail", "high"),
                texture_detail=cfg.get("texture_detail", "high"),
                target_faces=int(cfg.get("target_faces", 50000)),
                pedestal_shape=cfg.get("pedestal_shape", "round"),
                enable_rigging=bool(cfg.get("enable_rigging", False)),
                is_automated=True,
                metadata=metadata
            )
            database.update_statue_job(job_id, status="processing")

            # 3. Progress callback
            def on_progress(pct: int, step_name: str, step_idx: int, total_steps: int):
                metadata["progress"] = {
                    "pct": pct,
                    "step_name": step_name,
                    "step_idx": step_idx,
                    "total_steps": total_steps
                }
                database.update_statue_job(job_id, metadata=metadata)

            # 4. Execute pipeline
            res = self.statue_pipeline.process_statue(
                input_path=str(target_input),
                output_dir=str(job_work_dir),
                job_id=job_id,
                generator_type=cfg.get("generator", "trellis"),
                mesh_detail=cfg.get("mesh_detail", "high"),
                texture_detail=cfg.get("texture_detail", "high"),
                target_faces=int(cfg.get("target_faces", 50000)),
                pedestal_shape=cfg.get("pedestal_shape", "round"),
                enable_rigging=bool(cfg.get("enable_rigging", False)),
                progress_callback=on_progress
            )

            # 5. Publish to output directory if configured
            output_folder = Path(cfg.get("output_folder", str(STORAGE_DIR / "automation/output")))
            out_publish_dir = output_folder / job_id
            out_publish_dir.mkdir(parents=True, exist_ok=True)
            for fkey, fpath in res.get("files", {}).items():
                if Path(fpath).exists():
                    shutil.copyfile(fpath, str(out_publish_dir / Path(fpath).name))

            metadata["files"] = res.get("files", {})
            metadata["mesh_stats"] = res.get("mesh_stats", {})
            metadata["progress"] = {"pct": 100, "step_name": "Hoàn thành và sẵn sàng cho app tô tượng!", "step_idx": 5, "total_steps": 5}

            database.update_statue_job(
                job_id,
                status="completed",
                duration_sec=res.get("duration_sec", 0.0),
                num_vertices=res.get("mesh_stats", {}).get("num_vertices", 0),
                num_faces=res.get("mesh_stats", {}).get("num_faces", 0),
                num_parts=res.get("mesh_stats", {}).get("num_parts", 0),
                metadata=metadata
            )

            # 6. Dispatch Webhook
            webhook_url = cfg.get("webhook_url", "").strip()
            if webhook_url:
                self._dispatch_webhook(job_id, img_path.name, res, cfg, metadata)

            return res

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"❌ [Statue Automation] Failed job {job_id}:\n{tb}")
            database.update_statue_job(
                job_id,
                status="failed",
                error_message=str(e),
                metadata={"traceback": tb, "progress": {"pct": 100, "step_name": f"Lỗi: {e}"}}
            )
            return {"status": "failed", "error": str(e)}

        finally:
            with self._lock:
                self.active_jobs.pop(job_id, None)

    def _dispatch_webhook(
        self,
        job_id: str,
        filename: str,
        result: Dict[str, Any],
        cfg: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """Sends HTTP POST webhook with full model URLs and metadata to external consumer."""
        webhook_url = cfg.get("webhook_url", "").strip()
        if not webhook_url:
            return

        secret = cfg.get("webhook_secret", "").strip()
        retries = int(cfg.get("webhook_retry_count", 3))

        # Build accessible model URLs
        base_url = f"http://localhost:{self.server_port}"
        payload = {
            "event": "statue.completed",
            "job_id": job_id,
            "filename": filename,
            "timestamp": time.time(),
            "models": {
                "plaster_glb": f"{base_url}/api/statue/jobs/{job_id}/files/plaster_glb",
                "segmented_glb": f"{base_url}/api/statue/jobs/{job_id}/files/segmented_glb",
                "id_colored_glb": f"{base_url}/api/statue/jobs/{job_id}/files/id_colored_glb",
                "textured_glb": f"{base_url}/api/statue/jobs/{job_id}/files/textured_glb",
                "package_zip": f"{base_url}/api/statue/jobs/{job_id}/files/package_zip"
            },
            "mesh_stats": result.get("mesh_stats", {}),
            "settings": result.get("settings", {})
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "UniRig-3D-Statue-Automation/1.0",
            "X-Statue-Event": "statue.completed",
            "X-Statue-Job-ID": job_id
        }
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
            headers["X-Statue-Secret"] = secret

        webhook_success = False
        last_code = 0
        last_error = ""

        for attempt in range(1, retries + 1):
            try:
                print(f"📡 [Webhook] Dispatching to {webhook_url} (Attempt {attempt}/{retries})...")
                resp = requests.post(webhook_url, json=payload, headers=headers, timeout=15.0)
                last_code = resp.status_code
                if 200 <= resp.status_code < 300:
                    webhook_success = True
                    print(f"✅ [Webhook] Successfully delivered to {webhook_url} (HTTP {resp.status_code})")
                    break
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    print(f"⚠️ [Webhook] Server responded with error: {last_error}")
            except Exception as e:
                last_error = str(e)
                print(f"⚠️ [Webhook] Attempt {attempt} failed: {e}")
            
            time.sleep(2.0 * attempt)

        webhook_status = "sent" if webhook_success else "failed"
        metadata["webhook"] = {
            "status": webhook_status,
            "status_code": last_code,
            "url": webhook_url,
            "error": last_error if not webhook_success else None,
            "delivered_at": time.time() if webhook_success else None
        }
        database.update_statue_job(
            job_id,
            webhook_status=webhook_status,
            webhook_code=last_code,
            metadata=metadata
        )

def test_webhook_endpoint(url: str, secret: Optional[str] = None) -> Dict[str, Any]:
    """Tests a webhook URL by sending a sample ping payload."""
    test_payload = {
        "event": "statue.ping",
        "message": "Kiểm tra kết nối Webhook từ UniRig 3D Statue Painting Studio thành công!",
        "timestamp": time.time(),
        "test_sample_model": {
            "statue_id": "statue_test_sample",
            "status": "ready",
            "parts": ["Đầu", "Thân", "Tóc", "Tay", "Chân", "Đế tượng"]
        }
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "UniRig-3D-Statue-Automation/1.0",
        "X-Statue-Event": "statue.ping"
    }
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
        headers["X-Statue-Secret"] = secret

    t0 = time.time()
    try:
        r = requests.post(url, json=test_payload, headers=headers, timeout=10.0)
        t_elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "status": "success" if 200 <= r.status_code < 300 else "error",
            "status_code": r.status_code,
            "response_time_ms": t_elapsed,
            "response_body": r.text[:500]
        }
    except Exception as e:
        return {
            "status": "failed",
            "status_code": 0,
            "error": str(e),
            "response_time_ms": round((time.time() - t0) * 1000, 1)
        }
