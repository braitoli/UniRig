"""
Verification harness for the image-to-3D detail levels.

Checks the chosen levels survive the whole way down to the process that actually
runs the model -- the worker's HTTP payload for TRELLIS's fast path, and the argv
of the subprocess for TRELLIS's fallback and for Hunyuan3D. Reading the preset
table back would prove nothing; the values were hard-coded at those three call
sites, which is the bug this guards against.

No model runs: the transport is intercepted and a placeholder GLB stands in for
the generator's output.

Usage:
    python scripts/verify_detail_presets.py
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline import detail_presets  # noqa: E402
from pipeline.hunyuan3d_generator import Hunyuan3DImageTo3DGenerator  # noqa: E402
from pipeline.trellis_generator import TrellisImageTo3DGenerator  # noqa: E402

SCRATCH = REPO / "scratch" / "detail_presets_check"
IMAGE = REPO / "examples" / "sample_character.png"


def placeholder_glb(path: Path):
    """Something trimesh can load, so the generator's post-processing runs for real."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(1.0, 2.0, 1.0)).export(str(path))


class Captured:
    def __init__(self):
        self.payload: Optional[dict] = None
        self.argv: Optional[List[str]] = None

    def values(self) -> Dict[str, str]:
        """Everything the transport carried, as strings, for substring assertions."""
        if self.payload is not None:
            return {k: str(v) for k, v in self.payload.items()}
        out = {}
        for token in self.argv or []:
            if token.startswith("--") and "=" in token:
                key, value = token[2:].split("=", 1)
                out[key] = value
        return out


def run_trellis(mesh_level: str, texture_level: str, worker: bool) -> Captured:
    cap = Captured()
    out_glb = SCRATCH / f"trellis_{mesh_level}_{texture_level}_{int(worker)}.glb"
    kwargs = detail_presets.resolve("trellis", mesh_level, texture_level)

    def fake_get(url, *a, **k):
        if not worker:
            raise OSError("worker offline")
        return mock.Mock(status_code=200, json=lambda: {"status": "online"})

    def fake_post(url, json=None, **k):
        cap.payload = json
        placeholder_glb(Path(json["output_path"]))
        return mock.Mock(status_code=200, json=lambda: {"total_time_sec": 0.0})

    def fake_run(cmd, *a, **k):
        cap.argv = list(cmd)
        target = next(t.split("=", 1)[1] for t in cmd if t.startswith("--output_path="))
        placeholder_glb(Path(target))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("requests.get", fake_get), mock.patch("requests.post", fake_post), \
            mock.patch("subprocess.run", fake_run):
        TrellisImageTo3DGenerator().generate_3d_mesh(
            image_input=str(IMAGE), output_glb_path=str(out_glb), seed=1, **kwargs)
    return cap


def run_hunyuan(mesh_level: str, texture_level: str) -> Captured:
    cap = Captured()
    out_glb = SCRATCH / f"hunyuan_{mesh_level}_{texture_level}.glb"
    kwargs = detail_presets.resolve("hunyuan3d", mesh_level, texture_level)

    def fake_run(cmd, *a, **k):
        cap.argv = list(cmd)
        target = next(t.split("=", 1)[1] for t in cmd if t.startswith("--output_path="))
        placeholder_glb(Path(target))
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("subprocess.run", fake_run):
        Hunyuan3DImageTo3DGenerator().generate_3d_mesh(
            image_input=str(IMAGE), output_glb_path=str(out_glb), seed=1, **kwargs)
    return cap


# What each level must be visible as on the wire, keyed by the transport's field name.
TRELLIS_EXPECT = {
    ("preview", "preview"): {"resolution": "512", "decimation_target": "20000", "texture_size": "512", "tex_slat_steps": "6", "sparse_structure_steps": "6", "shape_slat_steps": "6"},
    ("standard", "standard"): {"resolution": "1024", "decimation_target": "150000", "texture_size": "2048", "tex_slat_steps": "15", "sparse_structure_steps": "12", "shape_slat_steps": "12"},
    ("high", "high"): {"resolution": "1024", "decimation_target": "300000", "texture_size": "4096", "tex_slat_steps": "30", "sparse_structure_steps": "12", "shape_slat_steps": "12"},
    ("preview", "high"): {"resolution": "512", "decimation_target": "20000", "texture_size": "4096", "tex_slat_steps": "30", "sparse_structure_steps": "6", "shape_slat_steps": "6"},
    ("high", "preview"): {"resolution": "1024", "decimation_target": "300000", "texture_size": "512", "tex_slat_steps": "6", "sparse_structure_steps": "12", "shape_slat_steps": "12"},
}
HUNYUAN_EXPECT = {
    ("preview", "preview"): {"octree_resolution": "128", "steps": "15", "paint_resolution": "512", "max_num_view": "6"},
    ("standard", "standard"): {"octree_resolution": "256", "steps": "30", "paint_resolution": "512", "max_num_view": "9"},
    ("high", "high"): {"octree_resolution": "256", "steps": "30", "paint_resolution": "768", "max_num_view": "9"},
    ("preview", "high"): {"octree_resolution": "128", "steps": "15", "paint_resolution": "768", "max_num_view": "9"},
}


def check(label: str, cap: Captured, expect: Dict[str, str]) -> List[str]:
    got = cap.values()
    if not got:
        return [f"{label}: nothing reached the transport"]
    return [f"{label}: {key} = {got.get(key)!r}, expected {value!r}"
            for key, value in expect.items() if got.get(key) != value]


def main() -> int:
    if not IMAGE.exists():
        print(f"[verify] missing test image {IMAGE}")
        return 1
    SCRATCH.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    checks = 0
    for (mesh, texture), expect in TRELLIS_EXPECT.items():
        for worker in (True, False):
            path = "worker" if worker else "subprocess"
            checks += 1
            failures += check(f"trellis/{path} mesh={mesh} texture={texture}",
                              run_trellis(mesh, texture, worker), expect)
    for (mesh, texture), expect in HUNYUAN_EXPECT.items():
        checks += 1
        failures += check(f"hunyuan3d mesh={mesh} texture={texture}",
                          run_hunyuan(mesh, texture), expect)

    # An unset level must reproduce exactly what the pipeline did before presets existed.
    default = detail_presets.resolve("trellis")
    if default != {"resolution": "1024", "decimate_target": 300_000, "texture_size": 4096,
                   "tex_slat_steps": 30, "sparse_structure_steps": 12, "shape_slat_steps": 12}:
        failures.append(f"trellis default drifted from the pre-preset values: {default}")
    checks += 1

    print(f"[verify] {checks} transport checks across "
          f"{len(TRELLIS_EXPECT)} TRELLIS and {len(HUNYUAN_EXPECT)} Hunyuan3D level pairs\n")
    for f in failures:
        print(f"  [FAIL] {f}")
    if not failures:
        print("  [PASS] every level reached the model process unchanged")
    print(f"\n[verify] {checks - len(failures)}/{checks} checks met")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
