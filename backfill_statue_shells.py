"""
Backfill các biến thể GLB có texture (textured / shell / shell_optimized) cho:
  1. Các statue job cũ đã chạy TRƯỚC khi tính năng shell export tồn tại  -> thiếu file => 404
  2. Các sample preset có statue_textured.glb bị mất material (0 texture) => model trắng

Nguồn chân lý: stage0_generated/*_generated_3d.glb (mesh gốc AI, còn nguyên UV + texture).
Script chạy được nhiều lần (idempotent): bỏ qua file đã có texture hợp lệ.
"""
import os, sys, json, struct, time, zipfile, shutil
from pathlib import Path

ROOT = Path("/home/braitoli/workspace/namnh/code/poc/UniRig")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "playground"))

import trimesh
from pipeline.statue_optimizer import (
    auto_ground_and_orient,
    extract_and_optimize_outer_shell,
    create_max_optimized_shell,
    optimize_material_textures,
)

FORCE = os.environ.get("FORCE_REBUILD") == "1"

JOBS_DIR = ROOT / "playground/storage/statue_jobs"
PRESETS_DIR = ROOT / "playground/static/sample_presets/models"


def glb_has_texture(path: Path) -> bool:
    """Đọc chunk JSON của GLB, kiểm tra có material + image + TEXCOORD_0 không."""
    if not path.exists():
        return False
    data = path.read_bytes()
    off, js = 12, None
    while off < len(data):
        clen, ctype = struct.unpack('<II', data[off:off + 8])
        if ctype == 0x4E4F534A:
            js = json.loads(data[off + 8:off + 8 + clen].decode('utf-8'))
            break
        off += 8 + clen
    if not js:
        return False
    if not js.get("materials") or not js.get("images"):
        return False
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "TEXCOORD_0" not in prim.get("attributes", {}):
                return False
    return True


def load_textured_source(src_glb: Path, orientation: str = "auto", reorient: bool = True):
    """Tái tạo đúng `textured_mesh` như statue_pipeline.process_statue làm.
    reorient=False khi nguồn là file shell/textured đã nằm sẵn trong hệ toạ độ cuối."""
    scene = trimesh.load(str(src_glb), process=False)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene.copy()
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    if getattr(mat, "baseColorTexture", None) is None:
        return None
    if reorient:
        mesh = auto_ground_and_orient(mesh, target_height=1.6, flatten_bottom=False, orientation=orientation)
    mesh.fix_normals()
    _ = mesh.vertex_normals
    return mesh


def export_glb(mesh, name: str, out: Path):
    data = trimesh.exchange.gltf.export_glb(
        trimesh.Scene({name: mesh}), include_normals=True
    )
    out.write_bytes(data)
    return len(data)


def rebuild(textured_mesh, out_dir: Path, stem: str, sep: str, log):
    """sep='_' cho job (<stem>_shell.glb); sep='' cho preset (statue_shell.glb)."""
    written = {}

    p_tex = out_dir / f"{stem}{sep}textured.glb"
    if glb_has_texture(p_tex) and not FORCE:
        log(f"    textured        : giữ nguyên (đã có texture)")
    else:
        optimize_material_textures(textured_mesh, max_texture_dim=4096)
        n = export_glb(textured_mesh, "Statue_Textured", p_tex)
        written["textured_glb"] = p_tex
        log(f"    textured        : {p_tex.name}  {n/1e6:.2f} MB")

    p_shell = out_dir / f"{stem}{sep}shell.glb"
    if glb_has_texture(p_shell) and not FORCE:
        log(f"    shell           : giữ nguyên (đã có texture)")
    else:
        shell = extract_and_optimize_outer_shell(textured_mesh, max_texture_dim=2048)
        n = export_glb(shell, "Statue_Outer_Shell", p_shell)
        written["shell_glb"] = p_shell
        log(f"    shell           : {p_shell.name}  {n/1e6:.2f} MB  ({len(shell.faces)} mặt)")

    p_opt = out_dir / f"{stem}{sep}shell_optimized.glb"
    if glb_has_texture(p_opt) and not FORCE:
        log(f"    shell_optimized : giữ nguyên (đã có texture)")
    else:
        opt = create_max_optimized_shell(textured_mesh, target_faces=45000, max_texture_dim=1536)
        n = export_glb(opt, "Statue_Shell_Max_Optimized", p_opt)
        written["shell_optimized_glb"] = p_opt
        log(f"    shell_optimized : {p_opt.name}  {n/1e6:.2f} MB  ({len(opt.faces)} mặt)")

    return written


def refresh_manifest_and_zip(job_dir: Path, stem: str, written: dict, log):
    manifest = job_dir / f"{stem}_manifest.json"
    if manifest.exists() and written:
        m = json.loads(manifest.read_text(encoding="utf-8"))
        m.setdefault("files", {}).update({k: v.name for k, v in written.items()})
        manifest.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"    manifest        : cập nhật {list(written.keys())}")

    zips = list(job_dir.glob("*_statue_package.zip"))
    if zips and written:
        pkg = zips[0]
        with zipfile.ZipFile(str(pkg), "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(job_dir.iterdir()):
                if f != pkg and f.is_file():
                    zf.write(str(f), arcname=f.name)
        log(f"    package_zip     : đóng gói lại {pkg.name}  {pkg.stat().st_size/1e6:.2f} MB")


def main():
    import database
    orient_by_job = {j["id"]: j.get("metadata", {}).get("orientation", "auto")
                     for j in database.list_statue_jobs(limit=200)}

    targets = []
    for d in sorted(JOBS_DIR.iterdir()):
        if d.is_dir():
            src = sorted(d.glob("stage0_generated/*.glb"))
            targets.append(("job", d, src[0] if src else None, orient_by_job.get(d.name, "auto")))

    # Preset -> job cung cấp mesh gốc có texture
    preset_source = {
        "cyber_turtle":   JOBS_DIR / "statue_1788406910_cyber_turtle/stage0_generated/cyber_turtle_generated_3d.glb",
        "mythical_beast": JOBS_DIR / "statue_1788417917_mythical_beast/stage0_generated/mythical_beast_generated_3d.glb",
    }
    for d in sorted(PRESETS_DIR.iterdir()):
        if d.is_dir():
            src = preset_source.get(d.name)
            targets.append(("preset", d, src if (src and src.exists()) else None, "auto"))

    total = len(targets)
    skipped, done = [], []
    t0 = time.time()

    for i, (kind, out_dir, src, orientation) in enumerate(targets, 1):
        print(f"\n[{i}/{total}] ({time.time()-t0:6.1f}s) {kind}: {out_dir.name}", flush=True)
        log = lambda s: print(s, flush=True)

        if kind == "job":
            stem = None
            for g in out_dir.glob("*_plaster.glb"):
                stem = g.name[:-len("_plaster.glb")]
            if stem is None:
                log("    -> BỎ QUA: không tìm thấy *_plaster.glb")
                skipped.append((out_dir.name, "không có plaster")); continue
            sep = "_"
        else:
            stem, sep = "statue", "_"

        already = all(glb_has_texture(out_dir / f"{stem}_{n}.glb")
                      for n in ("textured", "shell", "shell_optimized"))
        if already and not FORCE:
            log("    -> đã đủ 3 file có texture, bỏ qua")
            skipped.append((out_dir.name, "đã ổn")); continue

        reorient = True
        if src is None:
            # Không có mesh gốc: dùng chính file shell/textured đã có texture của thư mục này
            for cand in (out_dir / f"{stem}_textured.glb", out_dir / f"{stem}_shell.glb"):
                if glb_has_texture(cand):
                    src, reorient = cand, False
                    break
        if src is None:
            log("    -> BỎ QUA: không có nguồn nào còn texture")
            skipped.append((out_dir.name, "thiếu nguồn có texture")); continue

        log(f"    nguồn: {src.relative_to(ROOT)}  ({src.stat().st_size/1e6:.1f} MB, reorient={reorient})")
        mesh = load_textured_source(src, orientation, reorient)
        if mesh is None:
            log("    -> BỎ QUA: mesh gốc KHÔNG có texture (generator xuất mesh trắng)")
            skipped.append((out_dir.name, "mesh gốc không texture")); continue

        written = rebuild(mesh, out_dir, stem, sep, log)
        if kind == "job":
            refresh_manifest_and_zip(out_dir, stem, written, log)
        done.append((out_dir.name, list(written.keys())))

    print("\n" + "=" * 70)
    print(f"XONG sau {time.time()-t0:.1f}s — đã dựng lại {len(done)}, bỏ qua {len(skipped)}")
    for n, keys in done:
        print(f"  ✔ {n}: {', '.join(keys) if keys else '(không đổi)'}")
    for n, why in skipped:
        print(f"  – {n}: {why}")


if __name__ == "__main__":
    main()
