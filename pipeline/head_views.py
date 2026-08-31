"""
Multi-view rendering of a mesh's head region, shared by every 2D detector that needs
to find something on a face and map it back onto the 3D surface.

Two things here that a per-detector render loop does not give you:

* One OffscreenRenderer, one Scene and one camera node for the whole sweep. Creating a
  pyrender OffscreenRenderer allocates an EGL context, which dominates the cost of a
  512x512 render -- doing it per frame made a 24-angle sweep an order of magnitude more
  expensive than the renders themselves.
* Every view keeps its depth buffer and camera pose, so any 2D detection (box, polygon,
  landmark) can be backprojected onto the mesh surface with `backproject_pixels`.
"""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import trimesh


@dataclass
class View:
    """A single rendered look at the head, plus everything needed to invert it."""
    yaw_deg: float
    pitch_deg: float
    color: np.ndarray      # (S, S, 3) uint8 RGB
    depth: np.ndarray      # (S, S) float32, 0 where nothing was hit
    cam_pose: np.ndarray   # (4, 4) camera-to-world
    yfov: float
    image_size: int


def look_at_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world matrix for a camera at `eye` looking at `target` (OpenGL convention:
    the camera looks down its own -Z)."""
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def backproject_pixels(view: View, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Maps pixel coordinates to world-space points through the view's depth buffer.

    Pixels whose depth is 0 (background) are dropped, so the returned array can be
    shorter than the input. Returns (K, 3) float32.
    """
    px = np.asarray(px, dtype=np.int32)
    py = np.asarray(py, dtype=np.int32)
    s = view.image_size
    inside = (px >= 0) & (px < s) & (py >= 0) & (py < s)
    px, py = px[inside], py[inside]
    if len(px) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    d = view.depth[py, px]
    hit = d > 0
    px, py, d = px[hit], py[hit], d[hit]
    if len(px) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    f = (s / 2.0) / np.tan(view.yfov / 2.0)
    c = s / 2.0
    # Image row 0 is the top of the frame while camera +Y is up, hence the flipped Y.
    x_cam = (px - c) / f * d
    y_cam = (c - py) / f * d
    z_cam = -d  # camera looks down -Z, so a point `d` in front sits at Z = -d

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=1).astype(np.float32)
    return (pts_cam @ view.cam_pose.T)[:, :3].astype(np.float32)


def spiral_yaws(count: int) -> List[float]:
    """
    Yaw angles in degrees, evenly spaced over the full circle but ordered so that
    consecutive entries are far apart. Callers that stop early (once they have enough
    detections) then get angles spread around the head instead of one contiguous arc.
    """
    step = 360.0 / count
    order = []
    stride = max(1, count // 2 - (1 if count % 4 == 0 else 0))
    seen = set()
    idx = 0
    for _ in range(count):
        while idx % count in seen:
            idx += 1
        i = idx % count
        seen.add(i)
        order.append(i * step)
        idx += stride
    return order


class HeadViewRenderer:
    """
    Renders the mesh from arbitrary yaw/pitch angles around a head region, reusing a
    single EGL context and scene graph.

    Use as a context manager so the renderer is always released:

        with HeadViewRenderer(verts, faces, center, radius) as r:
            for v in r.sweep(yaws=[0, 45, 90]):
                ...
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        center: np.ndarray,
        radius: float,
        vertex_colors: Optional[np.ndarray] = None,
        image_size: int = 512,
        framing: float = 1.5,
    ):
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = float(radius)
        self.image_size = int(image_size)
        self.distance = self.radius * 2.8
        # Half the vertical frame covers `framing` head radii, so the head fills the
        # frame without its silhouette touching the border.
        self.yfov = float(2.0 * np.arctan2(self.radius * framing, self.distance))

        self._vertices = vertices
        self._faces = faces
        self._vertex_colors = vertex_colors
        self._renderer = None
        self._scene = None
        self._cam_node = None

    def __enter__(self) -> "HeadViewRenderer":
        import pyrender

        tmesh = trimesh.Trimesh(vertices=self._vertices, faces=self._faces, process=False)
        if self._vertex_colors is not None and len(self._vertex_colors) == len(self._vertices):
            tmesh.visual = trimesh.visual.color.ColorVisuals(tmesh, vertex_colors=self._vertex_colors)
        pymesh = pyrender.Mesh.from_trimesh(tmesh, smooth=True)

        # Ambient only, no directional light. The vertex colours are baked albedo, and the
        # 2D detectors that consume these frames are trained on flat illustration art:
        # adding a directional light blows out highlights and drags the render out of that
        # domain. Measured on a frontal chibi face, the anime eye detector's peak
        # confidence goes from 0.24 without the light to 0.00 with it.
        self._scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1.0], ambient_light=[1.0, 1.0, 1.0])
        self._scene.add(pymesh)
        camera = pyrender.PerspectiveCamera(yfov=self.yfov, aspectRatio=1.0)
        self._cam_node = self._scene.add(camera, pose=np.eye(4))
        self._renderer = pyrender.OffscreenRenderer(self.image_size, self.image_size)
        return self

    def __exit__(self, *exc) -> None:
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None
        self._scene = None
        self._cam_node = None
        return None

    def render(self, yaw_deg: float, pitch_deg: float = 0.0) -> View:
        """Renders one view. Yaw sweeps around +Y; positive pitch looks down from above."""
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        offset = np.array([
            self.distance * np.cos(pitch) * np.sin(yaw),
            self.distance * np.sin(pitch),
            self.distance * np.cos(pitch) * np.cos(yaw),
        ], dtype=np.float32)
        cam_pose = look_at_pose(self.center + offset, self.center)

        self._scene.set_pose(self._cam_node, cam_pose)
        color, depth = self._renderer.render(self._scene)

        return View(
            yaw_deg=float(yaw_deg),
            pitch_deg=float(pitch_deg),
            color=np.ascontiguousarray(color[:, :, :3]),
            depth=np.ascontiguousarray(depth),
            cam_pose=cam_pose,
            yfov=self.yfov,
            image_size=self.image_size,
        )

    def sweep(self, yaws: Sequence[float], pitches: Sequence[float] = (0.0,)) -> Iterator[View]:
        """Yields one View per (pitch, yaw) pair. Views are produced lazily so a caller
        can process and discard each frame instead of holding the whole sweep in memory."""
        for pitch in pitches:
            for yaw in yaws:
                yield self.render(yaw, pitch)
