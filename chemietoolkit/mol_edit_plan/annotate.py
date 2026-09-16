"""Overlay of the catalog's molecule boxes on the figure, for the edit-plan model.

Each ``[Mol]`` box of the catalog is drawn as a thin outline with a small numeric tag placed
just outside the box (above its top-left corner, or below when the box touches the top
edge). Tag N is molecule ``mol_00N`` of the catalog, so the model can point at a molecule
by looking at the picture instead of matching coordinates. Nothing is drawn inside a box
and the figure itself is not altered.
"""
import base64
import io

from PIL import Image, ImageDraw, ImageFont

OUTLINE = (0, 90, 220)          # blue, distinct from the black drawings and the red highlights papers use
TAG_BG = (0, 90, 220)
TAG_FG = (255, 255, 255)


def _font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def boxed_image(image_path, molecules, min_side=None):
    """PIL image of the figure with the catalog's molecule boxes and numeric tags.

    ``molecules``: the catalog's ``molecules`` list (``molecule_id`` = mol_NNN, ``bbox`` normalised
    x1, y1, x2, y2). Small figures are upscaled so the tags stay legible."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = 1.0
    if min_side and min(w, h) < min_side:
        scale = min_side / float(min(w, h))
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        w, h = img.size
    draw = ImageDraw.Draw(img)
    line = max(1, round(min(w, h) / 400))
    font = _font(max(10, round(min(w, h) / 55)))
    boxes = []
    for m in molecules:
        bbox = m.get("bbox")
        if bbox and len(bbox) == 4:
            boxes.append((m, (bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h)))

    def overlaps(r, others):
        return any(r[0] < o[2] and r[2] > o[0] and r[1] < o[3] and r[3] > o[1] for o in others)

    for m, (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=OUTLINE, width=line)
        tag = str(int(m["molecule_id"].split("_")[-1]))
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        pad = max(1, line)
        bw, bh = tw + 2 * pad, th + 2 * pad
        gap = line + 1
        # Tag positions, all outside the box: above-left, below-left, right of the top edge, left of
        # the top edge. The first one that stays on the image and covers no other molecule box wins,
        # so a tag does not hide a neighbouring structure or its counter-ion.
        candidates = [(x1, y1 - bh - gap), (x1, y2 + gap), (x2 + gap, y1), (x1 - bw - gap, y1)]
        others = [b for mm, b in boxes if mm is not m]
        chosen = None
        for tx, ty in candidates:
            if tx < 0 or ty < 0 or tx + bw > w or ty + bh > h:
                continue
            if not overlaps((tx, ty, tx + bw, ty + bh), others):
                chosen = (tx, ty)
                break
        if chosen is None:
            tx, ty = candidates[0] if candidates[0][1] >= 0 else candidates[1]
            chosen = (max(0, min(tx, w - bw)), max(0, min(ty, h - bh)))
        tx, ty = chosen
        draw.rectangle([tx, ty, tx + bw, ty + bh], fill=TAG_BG)
        draw.text((tx + pad, ty + pad), tag, fill=TAG_FG, font=font)
    return img


def boxed_image_base64(image_path, molecules, min_side=None):
    buf = io.BytesIO()
    boxed_image(image_path, molecules, min_side=min_side).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
