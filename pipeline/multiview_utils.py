"""
Multiview Utilities for Pixal3D Multi-View Pipeline.
Handles:
1. Auto-splitting turnaround sheets (1x4 horizontal or 2x2 grid) into 4 canonical views.
2. Packaging 4 separate uploaded views into standard Pixal3D directory structure with transforms.json.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
import numpy as np

# Pixal3D's camera convention, verified against the reference views the repo ships in
# assets/mv_images/example (world is Z-up, the character faces -Y):
#   azim000  camera at (0,-d,0)  the character looks straight at us
#   azim090  camera at (+d,0,0)  we see the character's LEFT side; its face points left in frame
#   azim180  camera at (0,+d,0)  we see its back
#   azim270  camera at (-d,0,0)  we see the character's RIGHT side; its face points right in frame
# A turnaround sheet is laid out Front / Right side / Back / Left side, so the second
# panel belongs to azim270 and the fourth to azim090. Filling the views in panel order
# instead swaps the two profiles, and the fused result comes out with the face turned
# the wrong way round.
PANEL_TO_VIEW_FILE = [
    "view00_azim000.png",  # panel 1: front
    "view03_azim270.png",  # panel 2: the character's right side
    "view02_azim180.png",  # panel 3: back
    "view01_azim090.png",  # panel 4: the character's left side
]

STANDARD_TRANSFORMS_4VIEWS = {
    "camera_angle_x": 0.3490658503988659,  # ~20 degrees FOV
    "mesh_scale": 1.0,
    "frames": [
        {
            "file_path": "view00_azim000.png",
            "name": "azim000",
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -3.1192049980163574],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        },
        {
            "file_path": "view01_azim090.png",
            "name": "azim090",
            "transform_matrix": [
                [0.0, 0.0, 1.0, 3.1192049980163574],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        },
        {
            "file_path": "view02_azim180.png",
            "name": "azim180",
            "transform_matrix": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 3.1192049980163574],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        },
        {
            "file_path": "view03_azim270.png",
            "name": "azim270",
            "transform_matrix": [
                [0.0, 0.0, -1.0, -3.1192049980163574],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
    ]
}


def _pad_and_center_to_square(img: Image.Image, target_size: int = 1024) -> Image.Image:
    """Centers an image and pads it into a square on white/transparent background."""
    mode = 'RGBA' if img.mode == 'RGBA' else 'RGB'
    bg_color = (255, 255, 255, 0) if mode == 'RGBA' else (255, 255, 255)
    
    w, h = img.size
    max_side = max(w, h)
    square = Image.new(mode, (max_side, max_side), bg_color)
    offset = ((max_side - w) // 2, (max_side - h) // 2)
    square.paste(img, offset)
    return square.resize((target_size, target_size), Image.Resampling.LANCZOS)


def _subject_alpha(img: Image.Image) -> Optional[np.ndarray]:
    """The image's own matte, or None when it carries no usable alpha."""
    if img.mode == 'RGBA':
        alpha = np.array(img.getchannel(3))
        if not np.all(alpha == 255):
            return alpha
    return None


def _subject_bbox(img: Image.Image, bg_rgb: np.ndarray, tol: float = 20.0) -> Tuple[int, int, int, int]:
    """
    Bounding box of the character, from its alpha when it has one and from the
    distance to the sheet's background colour otherwise. The alpha threshold matches
    the pipeline's own preprocess_image (alpha > 0.8 * 255).
    """
    alpha = _subject_alpha(img)
    if alpha is not None:
        mask = alpha > 0.8 * 255
    else:
        arr = np.array(img.convert('RGB'), dtype=np.float32)
        mask = np.linalg.norm(arr - bg_rgb, axis=-1) > tol
    ys, xs = np.where(mask)
    if len(ys) < 50:
        return (0, 0, img.width - 1, img.height - 1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _align_and_normalize_views(
    views: List[Image.Image],
    target_size: int = 1024,
    margin: float = 1.1
) -> List[Image.Image]:
    """
    Reframes the views into the framing transforms.json claims they have.

    All four frames share one camera_angle_x, one distance and one mesh_scale, so they
    are projections of the same object at the same scale -- which means they must be
    normalized with a SINGLE common scale, not one per view. Fitting each view's own
    height to a fixed fraction of the canvas is what breaks this: the 2D bbox of one
    object legitimately changes with the viewing angle (measured on the repo's own
    assets/mv_images/example: 0.826 / 0.845 / 0.888 / 0.845 of the frame height), and
    stretching each of them back to a common height rescales the geometry per view.

    So: one square crop window per view, centred on that view's bbox, all windows the
    same size -- the largest bbox side across the views times `margin`. The window and
    the 1.1 margin are the ones preprocess_image() uses on the single-view path, and
    centring on the bbox reproduces the reference views, whose subject centre sits at
    (0.4995, 0.5049) of the frame in all four.
    """
    first_arr = np.array(views[0].convert('RGB'), dtype=np.float32)
    bg_rgb = (first_arr[:15, :15].mean(axis=(0, 1)) + first_arr[:15, -15:].mean(axis=(0, 1))) / 2.0
    bg_tuple = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]))

    bboxes = [_subject_bbox(img, bg_rgb) for img in views]
    crop_size = max(max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in bboxes)
    crop_size = max(16, int(round(crop_size * margin)))

    processed = []
    for img, (x0, y0, x1, y1) in zip(views, bboxes):
        keep_alpha = _subject_alpha(img) is not None
        mode = 'RGBA' if keep_alpha else 'RGB'
        fill = (0, 0, 0, 0) if keep_alpha else bg_tuple

        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        left, top = int(round(cx - crop_size / 2.0)), int(round(cy - crop_size / 2.0))

        # Crop through a canvas rather than Image.crop so a window running past the
        # sheet edge is padded with the background instead of clamped, which would
        # shift the subject off-centre and break the shared scale.
        canvas = Image.new(mode, (crop_size, crop_size), fill)
        src = img.convert(mode)
        canvas.paste(src, (-left, -top))
        processed.append(canvas.resize((target_size, target_size), Image.Resampling.LANCZOS))

    return processed


def split_turnaround_sheet(
    sheet_path: str,
    output_dir: str,
    target_size: int = 1024,
    layout: str = "auto"
) -> str:
    """
    Takes a single turnaround character sheet and splits it into 4 canonical views:
    Front (0°), Right (90°), Back (180°), Left (270°).
    Uses adaptive projection-valley segmentation to handle varying figure widths.
    Writes transforms.json and the 4 image files into output_dir.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    im = Image.open(sheet_path).convert('RGB')
    w, h = im.size
    aspect = w / float(h)

    # Detect layout: 1x4 horizontal row typically has aspect >= 1.25 (e.g. 16:9 is 1.78)
    if layout == "auto":
        if aspect >= 1.25:
            layout = "1x4"
        else:
            layout = "2x2"

    filenames = PANEL_TO_VIEW_FILE

    slices = []
    if layout == "1x4":
        # Cut off bottom 10% (labels like "1. FRONT VIEW (0°)")
        content_h = int(h * 0.90) if h > 200 else h
        im_content = im.crop((0, 0, w, content_h))
        arr = np.array(im_content, dtype=np.float32)

        # Estimate background color
        bg = (arr[:15, :15].mean(axis=(0,1)) + arr[:15, -15:].mean(axis=(0,1))) / 2.0
        diff = np.linalg.norm(arr - bg, axis=-1)

        # 1D projection along columns
        proj_x = diff.sum(axis=0)
        k_size = max(5, int(w * 0.015))
        kernel = np.ones(k_size) / float(k_size)
        proj_smooth = np.convolve(proj_x, kernel, mode='same')

        # Find 3 valleys separating the 4 figures
        search_regions = [
            (int(w * 0.16), int(w * 0.35)),
            (int(w * 0.38), int(w * 0.62)),
            (int(w * 0.65), int(w * 0.84))
        ]
        cut_x = [0]
        for r_min, r_max in search_regions:
            r_min = max(0, min(w - 1, r_min))
            r_max = max(r_min + 1, min(w, r_max))
            valley_idx = r_min + np.argmin(proj_smooth[r_min:r_max])
            cut_x.append(int(valley_idx))
        cut_x.append(w)

        for i in range(4):
            slice_img = im.crop((cut_x[i], 0, cut_x[i+1], content_h))
            slices.append(slice_img)
    elif layout == "2x2":
        half_w = w // 2
        half_h = h // 2
        boxes = [
            (0, 0, half_w, half_h),           # Front
            (half_w, 0, w, half_h),           # Right
            (0, half_h, half_w, h),           # Back
            (half_w, half_h, w, h),           # Left
        ]
        for box in boxes:
            slices.append(im.crop(box))
    else:
        raise ValueError(f"Unsupported layout: {layout}. Use '1x4' or '2x2'")

    # Normalize views (align heights, centers, scales)
    normalized = _align_and_normalize_views(slices, target_size=target_size)
    for i, norm_img in enumerate(normalized):
        norm_img.save(str(out_path / filenames[i]))

    # Save standard transforms.json
    transforms_path = out_path / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(STANDARD_TRANSFORMS_4VIEWS, f, indent=2)

    return str(out_path)


def setup_multiview_from_4files(
    front_path: str,
    right_path: str,
    back_path: str,
    left_path: str,
    output_dir: str,
    target_size: int = 1024
) -> str:
    """
    Takes 4 individual view images and prepares the canonical Pixal3D views directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    inputs = list(zip([front_path, right_path, back_path, left_path], PANEL_TO_VIEW_FILE))

    raw_images = []
    for src_file, _ in inputs:
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Missing view image: {src_file}")
        img = Image.open(src_file)
        # Keep an uploaded PNG's own matte: it is cleaner than anything we would
        # recover from it later, and load_rgba() downstream uses it as-is.
        raw_images.append(img.convert('RGBA') if img.mode == 'RGBA' else img.convert('RGB'))

    normalized = _align_and_normalize_views(raw_images, target_size=target_size)
    for i, (_, dest_name) in enumerate(inputs):
        normalized[i].save(str(out_path / dest_name))

    transforms_path = out_path / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(STANDARD_TRANSFORMS_4VIEWS, f, indent=2)

    return str(out_path)
