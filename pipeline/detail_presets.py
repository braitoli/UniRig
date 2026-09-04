"""
Detail levels for image-to-3D generation.

Two knobs, deliberately independent: the diffusion grid the shape is solved on,
and the texture bake. They are worth separating because they do not cost the
same -- baking at 4096 was measured at ~360s of a 150-600s generation, so
dropping only the grid leaves a "preview" barely faster than the real thing,
while dropping only the texture keeps the silhouette exact.

Both default to `high`, which reproduces the values the pipeline used before
this module existed, so an unset level changes nothing.

TRELLIS and Hunyuan3D expose different knobs for the same idea, and the mapping
lives here rather than at each call site so the CLI, the HTTP server and the
playground cannot drift apart.
"""
from typing import Dict

LEVELS = ("preview", "standard", "high")
DEFAULT_LEVEL = "high"

# Accepted spellings of the generators
_HUNYUAN_ALIASES = ("hunyuan3d", "hunyuan", "hunyuan3d-2.1", "hy3d")
_PIXAL_ALIASES = ("pixal3d", "pixal", "tencentarc/pixal3d", "pixal-3d", "pixal3d_mv", "pixal_mv")

# The sampler step counts sit here rather than with the mesh sizes because diffusion is
# where a preview's time actually goes: the mesh and bake stage runs in ~2.6s of a ~45s
# preview, so decimate_target buys a lighter model, not a faster one.
_TRELLIS_MESH = {
    "preview": {"resolution": "512", "decimate_target": 20_000,
                "sparse_structure_steps": 6, "shape_slat_steps": 6},
    "standard": {"resolution": "1024", "decimate_target": 150_000,
                 "sparse_structure_steps": 12, "shape_slat_steps": 12},
    "high": {"resolution": "1024", "decimate_target": 300_000,
             "sparse_structure_steps": 12, "shape_slat_steps": 12},
}
# tex_slat_steps is a diffusion knob, not a bake knob, and it belongs here because it is
# what the texture actually costs. Measured interleaved on a loaded machine, the whole
# mesh+bake stage runs in ~2.6s of a ~45s preview -- decimation and texture size buy a
# lighter, softer model, not a faster one. Dropping these sampler steps from 30 to 6 is
# the only texture lever that moves the clock (49.4s -> 38.0s median).
_TRELLIS_TEXTURE = {
    "preview": {"texture_size": 512, "tex_slat_steps": 6},
    "standard": {"texture_size": 2048, "tex_slat_steps": 15},
    "high": {"texture_size": 4096, "tex_slat_steps": 30},
}

_HUNYUAN_MESH = {
    "preview": {"octree_resolution": 128, "num_steps": 15},
    "standard": {"octree_resolution": 256, "num_steps": 30},
    "high": {"octree_resolution": 256, "num_steps": 30},
}
# paint_resolution only accepts 512 or 768 (pipeline/hunyuan3d_infer.py), so the
# preview level buys its speed by painting fewer views rather than smaller ones.
_HUNYUAN_TEXTURE = {
    "preview": {"paint_resolution": 512, "max_num_view": 6},
    "standard": {"paint_resolution": 512, "max_num_view": 9},
    "high": {"paint_resolution": 768, "max_num_view": 9},
}

_PIXAL_MESH = {
    "preview": {"resolution": "1024", "low_vram": False, "ss_sampling_steps": 6,
                "shape_slat_sampling_steps": 6, "decimation_target": 100_000},
    "standard": {"resolution": "1024", "low_vram": False, "ss_sampling_steps": 12,
                 "shape_slat_sampling_steps": 12, "decimation_target": 300_000},
    "high": {"resolution": "1536", "low_vram": False, "ss_sampling_steps": 12,
             "shape_slat_sampling_steps": 12, "decimation_target": 1_000_000},
}
_PIXAL_TEXTURE = {
    "preview": {"texture_size": 1024, "tex_slat_steps": 6},
    "standard": {"texture_size": 2048, "tex_slat_steps": 12},
    "high": {"texture_size": 4096, "tex_slat_steps": 12},
}


def is_hunyuan(generator_type: str) -> bool:
    return str(generator_type or "").lower().strip() in _HUNYUAN_ALIASES


def is_pixal(generator_type: str) -> bool:
    return str(generator_type or "").lower().strip() in _PIXAL_ALIASES


def normalize(level) -> str:
    """A known level, falling back to the default rather than raising: these arrive
    from an HTTP form and a stored job record, and an unrecognised one should cost
    the caller its preset, not its job."""
    value = str(level or "").lower().strip()
    return value if value in LEVELS else DEFAULT_LEVEL


def resolve(generator_type: str, mesh_detail=None, texture_detail=None) -> Dict:
    """Keyword arguments to hand that generator's generate_3d_mesh()."""
    mesh, texture = normalize(mesh_detail), normalize(texture_detail)
    if is_hunyuan(generator_type):
        return {**_HUNYUAN_MESH[mesh], **_HUNYUAN_TEXTURE[texture]}
    if is_pixal(generator_type):
        return {**_PIXAL_MESH[mesh], **_PIXAL_TEXTURE[texture]}
    return {**_TRELLIS_MESH[mesh], **_TRELLIS_TEXTURE[texture]}
