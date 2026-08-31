"""
Semantic face parsing with `jonathandinu/face-parsing` (SegFormer fine-tuned on
CelebAMask-HQ), shared by every detector that needs to know which pixels of a rendered
head are eye, brow, nose or skin.

Why a segmenter and not a box detector: the consumer of this is an eyelid morph target.
A box drawn around an eye also contains cheek and brow, and a blink built from a box
folds whatever else the box caught. A per-pixel label gives the lid silhouette itself,
which is the shape the morph actually needs.

Two properties of the model matter downstream:

* It emits 19 labels in one pass, so asking for brows alongside eyes costs nothing extra.
  That is what makes the brow blendshape family (`browDown*`, `browInnerUp`,
  `browOuterUp*`) placeable instead of transferred blind.
* It is trained on photographs of human faces. On stylised or non-human heads it can
  return nothing at all, and callers must treat an empty result as "unknown", never as
  "no eyes here". `detect_eye_regions` honours that by returning None.
"""
from typing import Optional, Sequence, Tuple

import numpy as np

_MODEL_ID = "jonathandinu/face-parsing"

# CelebAMask-HQ label ids, in the order the model emits them.
LABEL_NAMES = (
    "background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
    "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r",
    "neck_l", "neck", "cloth",
)

EYE_LABELS = (4, 5)          # l_eye, r_eye
GLASSES_LABELS = (3,)        # eye_g
BROW_LABELS = (6, 7)         # l_brow, r_brow

# Eyes and glasses together are what localises an eye region. On stylised characters the
# parser routinely calls a large cartoon eye `eye_g` -- that label still sits exactly on
# the eye, so it is the right pixel set even though the name is wrong. Brows stay separate
# because they drive a different blendshape family.
EYE_REGION_LABELS = EYE_LABELS + GLASSES_LABELS

# Everything that belongs to a face rather than to the head or the body: skin, nose, eyes,
# brows, mouth and lips, but not hair, ears, neck or clothing. Used to work out where the
# face actually is before asking a finer question about a part of it -- the face labels
# cover thousands of pixels where an eye covers hundreds, so they survive a framing that
# an eye does not.
FACE_LABELS = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)

# A pixel the model is less sure than this about is treated as unclassified rather than
# taken at face value. Ported from the FaceParsing reference implementation, where the
# same threshold guards the same back-projection step.
MIN_CONFIDENCE = 0.60

_model = None
_processor = None
_device = None


def _load_model():
    """Loads the parser once, onto the GPU when there is one."""
    global _model, _processor, _device
    if _model is None:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = SegformerImageProcessor.from_pretrained(_MODEL_ID)
        _model = SegformerForSemanticSegmentation.from_pretrained(_MODEL_ID).eval().to(_device)
        print(f"[FaceParsing] Loaded {_MODEL_ID} on {_device}.")
    return _model, _processor, _device


def parse_faces(
    images: Sequence[np.ndarray],
    min_confidence: float = MIN_CONFIDENCE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Labels every pixel of a batch of RGB images.

    All images in a batch must share a resolution -- they always do here, because they
    come from one `HeadViewRenderer` sweep.

    Returns `(labels, confidence)`: labels is (B, H, W) uint8 holding a CelebAMask-HQ
    class id, with pixels below `min_confidence` forced to `background`; confidence is
    (B, H, W) float32 holding the winning class's softmax probability, left unmasked so a
    caller can inspect what was discarded.
    """
    import torch
    from PIL import Image

    if len(images) == 0:
        return np.zeros((0, 0, 0), np.uint8), np.zeros((0, 0, 0), np.float32)

    model, processor, device = _load_model()
    height, width = images[0].shape[:2]
    pils = [Image.fromarray(np.ascontiguousarray(img)) for img in images]
    inputs = processor(images=pils, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        # The model predicts at a quarter resolution; upsample before the argmax so a
        # label boundary lands on the pixel it belongs to rather than on a 4x4 block.
        logits = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        probs = torch.softmax(logits, dim=1)
        conf, labels = probs.max(dim=1)

    labels = labels.cpu().numpy().astype(np.uint8)
    conf = conf.cpu().numpy().astype(np.float32)
    labels[conf < min_confidence] = 0
    return labels, conf


def region_mask(labels: np.ndarray, label_ids: Sequence[int]) -> np.ndarray:
    """Boolean mask of the pixels carrying any of `label_ids`."""
    return np.isin(labels, np.asarray(label_ids, dtype=np.uint8))


def summarise(labels: np.ndarray) -> dict:
    """Pixel count per present label, keyed by name. For debug output only."""
    out = {}
    for lid, count in zip(*np.unique(labels, return_counts=True)):
        if 0 <= int(lid) < len(LABEL_NAMES):
            out[LABEL_NAMES[int(lid)]] = int(count)
    return out
