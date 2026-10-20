"""ChemEagle over a whole PDF: the article (and optionally its SI) handled as one document.

VisualHeist cuts every figure and table out of the pages, each crop goes through ``main.ChemEagle``
unchanged, and the results are put back into the document:

1. Document layer: the text layer (pdfplumber) in reading order, split into paragraphs with their section;
   every figure with its printed name (Scheme 2, Table S3), caption, footnotes, page and box; and per
   paragraph the figures it cites (Scheme 2, Table 1 entry 5) and the compound labels it names (bold 3a).
2. Compound labels across figures: one label -> structure table for the document, from the structures the
   molecule agent saw drawn next to each label, then from the compound names of characterisation paragraphs
   ("tert-Butyl Benzoyl-L-phenylalaninate (3a). ...", turned into structures by OPSIN), then from labelled
   molecules of the extracted reactions. A reactant, product or condition that only names a label gets that structure. When a figure
   does not draw a label but gives it a structure anyway (built from a template row), the structure drawn
   elsewhere replaces it; a figure that draws the label itself keeps its own, flagged if the two differ.
3. Text context: every figure gets its caption, footnotes and the paragraphs that cite it or name its
   compounds. Optionally (on by default) one text-only LLM call per figure adds the conditions, yields and
   selectivities the text states for its reactions. An item is kept only when its quoted evidence is found
   verbatim in the cited text; it is marked ``"source": "text"``.
4. Main text agent: before the extraction, every figure gets its role (substrate scope, optimization,
   mechanism, control experiment, spectra ...) from its caption and the sentences citing it; spectra and
   data plots are not extracted, and reactions of a mechanism are marked proposed_step. After it, the
   article and SI text are read for the reactions they state were run: a reaction a figure already holds
   gets the passage as a text_mention, a new one is added with ``"source": "text"``. Structures come from
   the label table or OPSIN, never from the LLM; every quote is checked against the cited paragraph.

    from main_pdf import ChemEagle_PDF
    doc = ChemEagle_PDF('paper.pdf', si_paths=['paper_si.pdf'], out_dir='paper_out')
    doc['reactions'], doc['figures'], doc['compounds'], doc['paragraphs']

    python main_pdf.py paper.pdf [--si paper_si.pdf] [--out paper_out] [--model gemini-3.7-flash]

main.py is imported, not changed: every figure goes through main.ChemEagle with the arguments given here.
Per-figure results are cached in ``out_dir/figures``, so an interrupted run resumes where it stopped.
"""
import argparse
import contextlib
import copy
import datetime
import json
import os
import re
import statistics
import sys
import time
import traceback
import unicodedata
from typing import List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

FIGURE_KINDS = {"scheme": "Scheme", "schemes": "Scheme", "table": "Table", "tables": "Table",
                "figure": "Figure", "figures": "Figure", "fig.": "Figure", "figs.": "Figure",
                "chart": "Chart", "charts": "Chart"}
CAPTION_RE = re.compile(r"^(Scheme|Table|Figure|Fig\.|Chart)\s+(S?\d{1,3})(\s*[.:|]|\s+(?=[A-Z(]))")
REF_RE = re.compile(r"\b(Schemes?|Tables?|Figures?|Figs?\.|Charts?)\s+"
                    r"(S?\d{1,3}[a-z]?(?:\s*(?:,|and|&|to|[-‐-―−])\s*S?\d{1,3}[a-z]?)*)")
ENTRY_RE = re.compile(r"\bentr(?:y|ies)\s+(\d{1,3}(?:\s*(?:,|and|&|to|[-‐-―−])\s*\d{1,3})*)", re.I)
BOLD_FONT_RE = re.compile(r"(bold|smbd|semibold|demibold|demi|black|heavy|,b$|-b$)", re.I)
LABEL_RE = re.compile(r"^(?:ent-|rac-)?(?:\d{1,3}[a-z]{0,3}|s\d{1,3}[a-z]{0,2}|[a-z]\d{1,3})'{0,2}$")
MARKER_RE = re.compile(r"^[a-z*†‡§](?:,[a-z])*$")
NO_LINK_SECTIONS = ("references", "notes", "associated content", "author information", "acknowledgment",
                    "acknowledgement", "supporting information")
CHEMICAL_ROLES = {"reagent", "catalyst", "solvent", "additive", "ligand", "base", "acid", "oxidant",
                  "reductant", "promoter", "initiator", "photocatalyst"}
CONDITION_ROLES = CHEMICAL_ROLES | {"temperature", "time", "atmosphere", "light", "pressure", "scale",
                                    "concentration", "other"}
RESULT_ROLES = {"yield", "ee", "er", "dr", "de", "conversion", "selectivity", "note"}


# ================================================================== text helpers
def norm_text(s):
    """Comparison form: NFKC, one dash, one quote style, no soft hyphens, single spaces, lower case."""
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = re.sub(r"[‐-―−﹘﹣－]", "-", s)
    s = re.sub(r"[‘’‚′]", "'", s)
    s = re.sub(r"[“”„″]", '"', s)
    s = s.replace("­", "")
    return re.sub(r"\s+", " ", s).strip().lower()


def norm_label(raw) -> Optional[str]:
    """A printed compound label in comparison form ('(3a)' -> '3a', "1a′" -> "1a'"), or None."""
    s = unicodedata.normalize("NFKC", str(raw or "")).strip()
    s = re.sub(r"^(compounds?|cpd\.?)\s+", "", s, flags=re.I)
    s = s.strip("()[]{}").rstrip(",.;:").strip("()[]{}")
    s = re.sub(r"[′’‘`´]", "'", s).replace("″", "''")
    s = re.sub(r"\s+", "", s).lower()
    return s if s and LABEL_RE.match(s) else None


def expand_label_token(token) -> List[str]:
    """Labels a bold token names: '3a' -> [3a]; '3a,b' -> [3a, 3b]; '3a−3d' / '3a-d' -> [3a .. 3d]."""
    t = unicodedata.normalize("NFKC", str(token or "")).strip().strip("()[]").rstrip(",.;:")
    t = re.sub(r"[‐-―−]", "-", t)
    m = re.match(r"^(\d{1,3})([a-z])\s*-\s*(?:\1)?([a-z])$", t)
    if m and m.group(2) < m.group(3) and ord(m.group(3)) - ord(m.group(2)) < 26:
        return [m.group(1) + chr(c) for c in range(ord(m.group(2)), ord(m.group(3)) + 1)]
    m = re.match(r"^(\d{1,3})([a-z](?:,[a-z])+)$", t)
    if m:
        return [m.group(1) + c for c in m.group(2).split(",")]
    lab = norm_label(t)
    return [lab] if lab else []


def _numbers(spec) -> List[str]:
    """'2-4' -> ['2', '3', '4']; '1 and 2' -> ['1', '2']; 'S1, S3' -> ['S1', 'S3']."""
    spec = re.sub(r"[‐-―−]", "-", spec)
    out = []
    for part in re.split(r"\s*(?:,|and|&)\s*", spec):
        m = re.match(r"^(S?)(\d{1,3})\s*(?:-|to)\s*(S?)(\d{1,3})$", part.strip())
        if m and int(m.group(4)) - int(m.group(2)) in range(1, 30):
            out += [m.group(1) + str(n) for n in range(int(m.group(2)), int(m.group(4)) + 1)]
        elif part.strip():
            out.append(re.sub(r"[a-z]$", "", part.strip()))
    return out


def _family(fontname):
    name = str(fontname or "").split("+")[-1]
    return re.split(r"[-,]", name)[0].lower()


def _is_bold(fontname):
    return bool(BOLD_FONT_RE.search(str(fontname or "").split("+")[-1]))


STYLE_HINT_RE = re.compile(r"(regular|roman|book|italic|oblique|light|medium|bold|smbd|semibold|demi|black|heavy|"
                           r"condensed|-it|it$|mt$)", re.I)
LABELISH_RE = re.compile(r"^\(?\d{1,3}[a-z]{0,3}\)?[,.;:]?$")


def mark_emphasis(words):
    """Set w["bold"] on every word. Fonts named bold count; so does a full-size font whose name has no style
    hint (subset aliases such as AdvOT51c1769e) when label-like tokens (1a, 2b) make up a quarter of it."""
    if not words:
        return
    typical = statistics.median([w["size"] for w in words])
    stats = {}
    for w in words:
        if w["size"] >= 0.85 * typical:
            st = stats.setdefault(w["fontname"], [0, 0])
            st[0] += 1
            st[1] += bool(LABELISH_RE.match(w["text"]))
    emphasis = {f for f in stats if _is_bold(f)}
    for f, (n, labelish) in stats.items():
        if f not in emphasis and not STYLE_HINT_RE.search(f.split("+")[-1]) and labelish >= 3 and labelish >= 0.25 * n:
            emphasis.add(f)
    for w in words:
        w["bold"] = w["fontname"] in emphasis or _is_bold(w["fontname"])


# ================================================================== page text layout
def _rows(words):
    """Words grouped into visual rows. A word joins a row when its box overlaps the row's core (the box of
    its main-size text) by half the smaller height, so sub- and superscripts and footnote markers stay with
    their line while the next line, which does not overlap, starts a new row."""
    if not words:
        return []

    def share(row, w):
        overlap = min(row["bottom"], w["bottom"]) - max(row["top"], w["top"])
        return overlap / max(1e-6, min(w["bottom"] - w["top"], row["bottom"] - row["top"]))

    def beside(row, w, gap=12.0):
        return any(max(w["x0"], v["x0"]) - min(w["x1"], v["x1"]) <= gap for v in row["words"])

    typical = statistics.median([w["size"] for w in words])
    rows = []
    # full-size text first: it defines the lines
    for w in sorted((w for w in words if w["size"] >= 0.8 * typical), key=lambda w: (w["top"], w["x0"])):
        best = max(rows[-8:], key=lambda r: share(r, w), default=None)
        if best is None or share(best, w) < 0.5:
            rows.append({"top": w["top"], "bottom": w["bottom"], "size": w["size"], "words": [w]})
        else:
            best["words"].append(w)
    # then sub- and superscripts and footnote markers, each to the overlapping line it sits next to
    for w in sorted((w for w in words if w["size"] < 0.8 * typical), key=lambda w: (w["top"], w["x0"])):
        near = [r for r in rows if share(r, w) >= 0.3 and beside(r, w)]
        best = max(near, key=lambda r: share(r, w), default=None)
        if best is None:
            rows.append({"top": w["top"], "bottom": w["bottom"], "size": w["size"], "words": [w]})
        else:
            best["words"].append(w)
    return sorted(rows, key=lambda r: r["top"])


def _clean_words(words):
    """Upright text only (no rotated margin notes), without oversized glyphs (section squares, drop caps)."""
    words = [w for w in words if w.get("upright", True) and w["text"].strip()]
    if not words:
        return words
    typical = statistics.median([w["size"] for w in words])
    return [w for w in words if not (w["size"] > 2.5 * typical and len(w["text"]) <= 2)]


def _segment(words):
    words = sorted(words, key=lambda w: w["x0"])
    text = ""
    for i, w in enumerate(words):
        if i and w["x0"] - words[i - 1]["x1"] > 0.8:
            text += " "
        text += w["text"]
    sizes = [w["size"] for w in words]
    nchar = sum(len(w["text"]) for w in words) or 1
    fam = {}
    for w in words:
        fam[_family(w["fontname"])] = fam.get(_family(w["fontname"]), 0) + len(w["text"])
    return {"words": words, "text": text, "x0": min(w["x0"] for w in words), "x1": max(w["x1"] for w in words),
            "top": min(w["top"] for w in words), "bottom": max(w["bottom"] for w in words),
            "size": statistics.median(sizes),
            "bold": sum(len(w["text"]) for w in words if w.get("bold", _is_bold(w["fontname"]))) / nchar,
            "family": max(fam, key=fam.get)}


def page_segments(words, gutter=12.0):
    """Rows split at wide horizontal gaps: one segment per column piece of a line."""
    out = []
    for row in _rows(words):
        ws = sorted(row["words"], key=lambda w: w["x0"])
        start = 0
        for i in range(1, len(ws) + 1):
            if i == len(ws) or ws[i]["x0"] - ws[i - 1]["x1"] > gutter:
                out.append(_segment(ws[start:i]))
                start = i
    return out


def reading_order(segs, page_width):
    """Segments in reading order: full-width bands top to bottom, two-column bands left then right."""
    mid = page_width / 2
    wide = [s for s in segs if s["x1"] - s["x0"] > 0.25 * page_width]
    two_col = (sum(1 for s in wide if s["x1"] <= mid + 4) >= 4 and sum(1 for s in wide if s["x0"] >= mid - 4) >= 4)
    for s in segs:
        s["col"] = ("L" if s["x1"] <= mid + 4 else "R" if s["x0"] >= mid - 4 else "S") if two_col else "S"
    bands = []
    for s in sorted(segs, key=lambda s: s["top"]):
        kind = "span" if s["col"] == "S" else "cols"
        if not bands or bands[-1]["kind"] != kind:
            bands.append({"kind": kind, "segs": []})
        bands[-1]["segs"].append(s)
        s["band"] = len(bands) - 1
    ordered = []
    for b in bands:
        if b["kind"] == "span":
            ordered += sorted(b["segs"], key=lambda s: s["top"])
        else:
            ordered += sorted([s for s in b["segs"] if s["col"] == "L"], key=lambda s: s["top"])
            ordered += sorted([s for s in b["segs"] if s["col"] == "R"], key=lambda s: s["top"])
    return ordered


def _is_heading(seg):
    t = seg["text"].strip()
    return bool(t) and len(t) < 100 and bool(re.search(r"[A-Za-z]{3}", t)) and (t.startswith("■") or (seg["bold"] >= 0.85 and not t.endswith((".", ","))
                                                                      and not CAPTION_RE.match(t)))


def _join(a, b):
    if re.search(r"[A-Za-z]-$", a) and re.match(r"^[a-z]", b):
        return a[:-1] + b
    return a + " " + b


# ================================================================== figures
def _caption_and_footnotes(fsegs, body_family):
    """Caption, footnotes and other printed text of one figure box (segments inside the box)."""
    fsegs = sorted(fsegs, key=lambda s: (s["top"], s["x0"]))
    caption, cap_idx = None, set()
    for i, s in enumerate(fsegs):
        m = CAPTION_RE.match(s["text"].strip())
        if m and (s["bold"] > 0.2 or m.group(3).strip()):
            caption, cap_idx = s["text"].strip(), {i}
            last = s
            for j in range(i + 1, len(fsegs)):
                t = fsegs[j]
                if (t["top"] - last["bottom"] < 0.9 * last["size"] and t["family"] == s["family"]
                        and not _marker(t) and t["x0"] < last["x1"] and t["x1"] > s["x0"]):
                    caption = _join(caption, t["text"].strip())
                    cap_idx.add(j)
                    last = t
                elif t["top"] - last["bottom"] >= 0.2 * last["size"]:
                    break
            break
    footnotes, other = _footnotes([s for i, s in enumerate(fsegs) if i not in cap_idx and s["family"] == body_family])
    return caption, footnotes, other


def _footnotes(segs):
    """Footnotes ('aReaction conditions: ...') among text segments, and the segments that are none."""
    footnotes, other, current = [], [], None
    for s in sorted(segs, key=lambda s: (s["top"], s["x0"])):
        mk = _marker(s)
        if mk:
            current = {"marker": mk, "text": s["text"].strip()[len(mk):].strip(), "bottom": s["bottom"]}
            footnotes.append(current)
        elif current is not None and s["top"] - current["bottom"] < 0.9 * s["size"]:
            current["text"] = _join(current["text"], s["text"].strip())
            current["bottom"] = s["bottom"]
        else:
            current = None
            other.append(s["text"].strip())
    for k, fn in enumerate(footnotes):
        fn["id"] = "fn_%s" % (re.sub(r"[^a-z0-9]", "", fn["marker"]) or str(k + 1))
        fn.pop("bottom", None)
    return footnotes, other


def _marker(seg):
    """The footnote marker a segment starts with ('a' of 'aReaction conditions'), or None."""
    ws = seg["words"]
    if len(ws) < 2:
        return None
    first = ws[0]
    rest = statistics.median([w["size"] for w in ws[1:]])
    if MARKER_RE.match(first["text"]) and first["size"] <= 0.85 * rest:
        return first["text"]
    return None


def _figure_name(caption, doc_tag):
    m = CAPTION_RE.match(caption or "")
    if not m:
        return None, None, None
    kind = FIGURE_KINDS.get(m.group(1).lower(), m.group(1).title())
    return "%s %s" % (kind, m.group(2)), kind, m.group(2)


def _paragraph_links(para, figures_by_name, doc_of):
    refs, last_table = [], None
    text = para["text"]
    for m in REF_RE.finditer(text):
        kind = FIGURE_KINDS.get(m.group(1).lower(), None)
        if not kind:
            continue
        for num in _numbers(m.group(2)):
            name = "%s %s" % (kind, num)
            ids = figures_by_name.get(name, [])
            ids = [i for i in ids if doc_of[i] == para["doc"]] or ids   # the article's Scheme 1, not the SI's
            entries = []
            tail = text[m.end():m.end() + 40]
            em = ENTRY_RE.match(tail.lstrip(" ,;("))
            if kind == "Table" and em:
                entries = _numbers(em.group(1))
            refs.append({"figure": name, "figure_ids": ids, "entries": entries})
            if kind == "Table":
                last_table = refs[-1]
    for em in ENTRY_RE.finditer(text):   # "(entries 6 and 7)" without a table name: the last table cited
        before = text[max(0, em.start() - 45):em.start()]
        if last_table is not None and not re.search(r"Tables?\s+S?\d", before):
            for e in _numbers(em.group(1)):
                if e not in last_table["entries"]:
                    last_table["entries"].append(e)
    return refs


# ================================================================== document layer
def _detector(large):
    """VisualHeist (pdfmodel.methods), on the GPU when there is one."""
    import torch
    from pdfmodel.methods import BASE_MODEL_ID, LARGE_MODEL_ID, _create_model
    model, processor = _create_model(LARGE_MODEL_ID, "large") if large else _create_model(BASE_MODEL_ID, "base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), processor


def _detect(image, model, processor):
    """pdfmodel.methods._tf_id_detection with the inputs on the model's device (same prompt and decoding)."""
    import torch
    inputs = processor(text="<OD>", images=image, return_tensors="pt")
    device = next(model.parameters()).device
    with torch.no_grad():
        ids = model.generate(input_ids=inputs["input_ids"].to(device), pixel_values=inputs["pixel_values"].to(device),
                             max_new_tokens=1024, do_sample=False, num_beams=3)
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(text, task="<OD>", image_size=(image.width, image.height))["<OD>"]


def parse_document(files: Sequence, out_dir: str, *, dpi: int = 200, large_detector: bool = True,
                   detector=None) -> dict:
    """Pages, paragraphs and figures of the PDFs in ``files`` = [(doc_tag, path), ...]; crops are saved to
    out_dir/crops. Deterministic and offline (no LLM)."""
    import pdfplumber
    from pdf2image import convert_from_path
    loaded = []

    def detect(image):
        if not loaded:
            loaded.append(detector or _detector(large_detector))
        return _detect(image, *loaded[0])

    crop_dir = os.path.join(out_dir, "crops")
    os.makedirs(crop_dir, exist_ok=True)
    det_dir = os.path.join(out_dir, "detections")
    os.makedirs(det_dir, exist_ok=True)
    doc = {"files": [], "figures": [], "paragraphs": []}
    for doc_tag, path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        pdf = pdfplumber.open(path)
        pages = []
        all_words = []
        for pno, page in enumerate(pdf.pages, start=1):
            words = _clean_words(page.extract_words(x_tolerance=1.5, y_tolerance=2.5, keep_blank_chars=False,
                                                    extra_attrs=["fontname", "size"]))
            pages.append({"page": pno, "width": float(page.width), "height": float(page.height), "words": words})
            all_words += words
        mark_emphasis(all_words)
        fam = {}
        for w in all_words:
            fam[_family(w["fontname"])] = fam.get(_family(w["fontname"]), 0) + len(w["text"])
        body_family = max(fam, key=fam.get) if fam else ""
        doc["files"].append({"doc": doc_tag, "path": os.path.abspath(path), "pages": len(pages),
                             "body_font": body_family})

        # ---- figures: VisualHeist boxes, crops, captions and footnotes
        figs_here, consumed = [], {}
        det_cache = os.path.join(det_dir, "%s_%s_dpi%d_%s.json" % (doc_tag, stem, dpi, "large" if large_detector else "base"))
        detections = json.load(open(det_cache, encoding="utf-8")) if os.path.exists(det_cache) else {}
        for pg in pages:
            image = convert_from_path(path, dpi=dpi, first_page=pg["page"], last_page=pg["page"])[0]
            if str(pg["page"]) not in detections:
                found = detect(image)
                detections[str(pg["page"])] = {"bboxes": [[float(v) for v in b] for b in found.get("bboxes") or []],
                                               "labels": list(found.get("labels") or [])}
                json.dump(detections, open(det_cache, "w", encoding="utf-8"), indent=1)
            ann = detections[str(pg["page"])]
            sx, sy = pg["width"] / image.width, pg["height"] / image.height
            for k, (box, det_label) in enumerate(zip(ann.get("bboxes") or [], ann.get("labels") or [])):
                x1, y1, x2, y2 = [float(v) for v in box]
                fid = "%s_p%d_%d" % (doc_tag, pg["page"], k + 1)
                img_path = os.path.join(crop_dir, "%s_image_%d_%d.png" % (stem, pg["page"], k + 1))
                image.crop((x1, y1, x2, y2)).save(img_path)
                bbox = [round(x1 * sx, 1), round(y1 * sy, 1), round(x2 * sx, 1), round(y2 * sy, 1)]
                inside = [w for w in pg["words"]
                          if bbox[0] - 3 <= (w["x0"] + w["x1"]) / 2 <= bbox[2] + 3
                          and bbox[1] - 3 <= (w["top"] + w["bottom"]) / 2 <= bbox[3] + 3]
                caption, footnotes, other = _caption_and_footnotes(page_segments(inside), body_family)
                figs_here.append({"figure_id": fid, "doc": doc_tag, "page": pg["page"], "bbox": bbox,
                                  "detector_label": det_label, "image": img_path, "caption": caption,
                                  "footnotes": footnotes, "box_text": "\n".join(other)[:4000]})
                for w in inside:
                    consumed[id(w)] = fid
            del image

        # ---- body text: everything outside the figure boxes, minus running headers and footers
        segs_by_page = {}
        margin_text = {}

        def in_margin(s, pg):
            return s["top"] < 0.09 * pg["height"] or s["bottom"] > 0.93 * pg["height"]

        for pg in pages:
            body = [w for w in pg["words"] if id(w) not in consumed]
            segs = reading_order(page_segments(body), pg["width"])
            for s in segs:
                s["page"] = pg["page"]
                if in_margin(s, pg):
                    key = re.sub(r"\d+", "#", s["text"].lower()).strip()
                    margin_text[key] = margin_text.get(key, 0) + 1
            segs_by_page[pg["page"]] = segs
        for pg in pages:
            kept = []
            for s in segs_by_page[pg["page"]]:
                key = re.sub(r"\d+", "#", s["text"].lower()).strip()
                if in_margin(s, pg) and (margin_text.get(key, 0) >= 2 or re.fullmatch(r"[#\s.\-]*", key)
                                         or "doi.org" in key or "pubs.acs.org" in key):
                    continue
                kept.append(s)
            segs_by_page[pg["page"]] = kept

        def overlaps(s, box, share=0.3):
            return min(s["x1"], box[2]) - max(s["x0"], box[0]) > share * min(s["x1"] - s["x0"], box[2] - box[0])

        # footnotes printed just below their box ('aReaction conditions: ...' under a scheme)
        for f in figs_here:
            segs = sorted((s for s in segs_by_page[f["page"]] if overlaps(s, f["bbox"]) and s["top"] >= f["bbox"][3] - 2),
                          key=lambda s: s["top"])
            if not segs or segs[0]["top"] > f["bbox"][3] + 25 or not _marker(segs[0]):
                continue
            chain = [segs[0]]
            for s in segs[1:]:
                if s["top"] - chain[-1]["bottom"] < 0.9 * s["size"] and not _is_heading(s):
                    chain.append(s)
                elif s["top"] > chain[-1]["bottom"]:
                    break
            notes, _ = _footnotes(chain)
            if notes:
                known = {fn["id"] for fn in f["footnotes"]}
                f["footnotes"] += [fn for fn in notes if fn["id"] not in known]
                for s in chain:
                    s["_footnote_of"] = f["figure_id"]

        # captions printed outside their box (above or below it)
        for f in figs_here:
            if f["caption"]:
                continue
            near = [s for s in segs_by_page[f["page"]]
                    if (f["bbox"][1] - 45 <= s["bottom"] <= f["bbox"][1] + 2 or f["bbox"][3] - 2 <= s["top"] <= f["bbox"][3] + 45)
                    and min(s["x1"], f["bbox"][2]) - max(s["x0"], f["bbox"][0]) > 0.3 * (f["bbox"][2] - f["bbox"][0])]
            for s in sorted(near, key=lambda s: s["top"]):
                if CAPTION_RE.match(s["text"].strip()):
                    f["caption"] = s["text"].strip()
                    s["_caption_of"] = f["figure_id"]
                    break
        for f in figs_here:
            f["name"], f["kind"], f["number"] = _figure_name(f["caption"], doc_tag)
        doc["figures"] += figs_here

        # ---- paragraphs
        paras, cur, prev = [], None, None
        section = ""
        col_left = {}
        for pg in pages:
            for s in segs_by_page[pg["page"]]:
                key = (pg["page"], s["band"], s["col"])
                col_left[key] = min(col_left.get(key, 1e9), s["x0"])
        for pg in pages:
            for s in segs_by_page[pg["page"]]:
                if s.get("_caption_of") or s.get("_footnote_of"):
                    continue
                heading = _is_heading(s)
                key = (pg["page"], s["band"], s["col"])
                new = (cur is None or heading or cur.get("heading") or prev is None
                       or (s["page"], s["band"], s["col"]) != (prev["page"], prev["band"], prev["col"])
                       or s["top"] - prev["bottom"] > 0.9 * prev["size"]
                       or s["x0"] > col_left[key] + 0.7 * s["size"])
                if new:
                    if heading:
                        section = re.sub(r"^■\s*", "", s["text"].strip()).strip()
                    cur = {"doc": doc_tag, "pages": [s["page"]], "section": section, "heading": heading,
                           "text": s["text"].strip(), "words": list(s["words"])}
                    paras.append(cur)
                else:
                    cur["text"] = _join(cur["text"], s["text"].strip())
                    cur["words"] += s["words"]
                    if s["page"] not in cur["pages"]:
                        cur["pages"].append(s["page"])
                prev = s
        merged = []   # a paragraph broken by a column or page change continues in lower case
        for p in paras:
            if (merged and not p["heading"] and not merged[-1]["heading"] and re.match(r"^[a-z(]", p["text"])
                    and not re.search(r"[.!?:]$", merged[-1]["text"])):
                merged[-1]["text"] = _join(merged[-1]["text"], p["text"])
                merged[-1]["words"] += p["words"]
                merged[-1]["pages"] += [x for x in p["pages"] if x not in merged[-1]["pages"]]
            else:
                merged.append(p)
        for p in merged:
            if p["heading"]:
                continue
            lead = []
            for w in p["words"]:
                if w["bold"]:
                    lead.append(w["text"])
                else:
                    break
            p["lead"] = " ".join(lead) if len(lead) >= 2 else ""
            labels = []
            for w in p["words"]:
                if w["bold"]:
                    for lab in expand_label_token(w["text"]):
                        if lab not in labels:
                            labels.append(lab)
            p["labels"] = labels
        doc["paragraphs"] += [p for p in merged if not p["heading"]]

    # ---- ids and links (after every file is read: the article cites SI figures and the other way round)
    by_name, doc_of = {}, {f["figure_id"]: f["doc"] for f in doc["figures"]}
    for f in doc["figures"]:
        if f["name"]:
            by_name.setdefault(f["name"], []).append(f["figure_id"])
    for k, p in enumerate(doc["paragraphs"]):
        p["id"] = "%s:p%d" % (p["doc"], k + 1)
        p.pop("heading", None)
        p.pop("words", None)
        linkable = not any(x in p["section"].lower() for x in NO_LINK_SECTIONS)
        p["figure_refs"] = _paragraph_links(p, by_name, doc_of) if linkable else []
        if not linkable:
            p["labels"] = []
    for f in doc["figures"]:
        f["cited_by"] = [p["id"] for p in doc["paragraphs"]
                         if any(f["figure_id"] in r["figure_ids"] for r in p["figure_refs"])]
    return doc


# ================================================================== per-figure extraction
@contextlib.contextmanager
def _in_repo():
    old = os.getcwd()
    os.chdir(_HERE)
    try:
        yield
    finally:
        os.chdir(old)


def run_figure(fig, out_dir, chemeagle_kwargs, resume=True):
    """main.ChemEagle on one crop, cached as out_dir/figures/<figure_id>.json (log in out_dir/logs)."""
    cache = os.path.join(out_dir, "figures", fig["figure_id"] + ".json")
    if resume and os.path.exists(cache):
        rec = json.load(open(cache, encoding="utf-8"))
        if rec.get("status") == "ok":
            return rec
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    import get_molecular_agent
    import main
    art, rec, t0 = {}, {"figure_id": fig["figure_id"], "image": fig["image"]}, time.time()
    with open(os.path.join(out_dir, "logs", fig["figure_id"] + ".log"), "w", encoding="utf-8", errors="replace") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            rec["output"] = main.ChemEagle(fig["image"], artifacts=art, **chemeagle_kwargs)
            rec["status"] = "ok"
        except Exception as exc:
            rec["status"], rec["error"] = "error", "%s: %s" % (type(exc).__name__, str(exc)[:400])
            traceback.print_exc()
    rec["seconds"] = round(time.time() - t0, 1)
    rec["label_structures"] = get_molecular_agent.label_structures(fig["image"])
    rec["agents"] = art.get("ordered_agents")
    rec["failed_agents"] = art.get("failed_agents")
    json.dump(rec, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return rec


def _reactions(output):
    return [r for r in (output or {}).get("reactions") or [] if isinstance(r, dict)] if isinstance(output, dict) else []


# ================================================================== compound labels across figures
def _mol(smiles):
    from rdkit import Chem
    if not isinstance(smiles, str) or not smiles.strip() or smiles.strip().lower() in ("none", "null", "n/a"):
        return None
    return Chem.MolFromSmiles(smiles)


def _canon(smiles, stereo=True):
    from rdkit import Chem
    m = _mol(smiles)
    if m is None:
        return None
    if not stereo:
        Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def _skeleton(smiles):
    """Largest fragment without charges, explicit hydrogens, isotopes or stereo: '[NH3+]...[Cl-]' and 'N...'
    are the same compound written as salt or free base."""
    from rdkit import Chem
    m = _mol(smiles)
    if m is None:
        return None
    m = Chem.RWMol(max(Chem.GetMolFrags(m, asMols=True), key=lambda f: f.GetNumHeavyAtoms()))
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(False)
        a.SetIsotope(0)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return Chem.MolToSmiles(m, isomericSmiles=False)


def _concrete(smiles, min_atoms=5):
    m = _mol(smiles)
    return m is not None and "*" not in smiles and m.GetNumHeavyAtoms() >= min_atoms


NAMED_RE = re.compile(r"^(.{3,300}?)\s\(([^()]{1,12})\)[.:](?:\s|$)")   # the name may be a paragraph of its own
UNIT_LABEL_RE = re.compile(r"^\d+(?:\.\d+)?(?:ml|mg|kg|mmol|mol|min|nm|μl|ul)$")   # not g, h, l: those are label letters
CHARACTERISATION_RE = re.compile(r"NMR|HRMS|\bIR\b|m/z|\bmp\b|\bRf\b|yield|solid|\boil\b", re.I)


def named_compounds(doc, out_dir=None):
    """label -> {name, smiles, paragraph}: the compounds the text names in characterisation paragraphs
    ("tert-Butyl Benzoyl-L-phenylalaninate (3a). Compound 3a was purified ..."), turned into structures by
    OPSIN (the pipeline's resolver: the web service, else the bundled jar). Cached in out_dir/named_compounds.json."""
    from chemietoolkit import helper
    java_dir = os.path.join(sys.prefix, "Library", "bin")   # a conda env ships its own Java for local OPSIN
    if os.path.exists(os.path.join(java_dir, "java.exe")) and java_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = java_dir + os.pathsep + os.environ.get("PATH", "")
    cache_path = os.path.join(out_dir, "named_compounds.json") if out_dir else None
    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path and os.path.exists(cache_path) else {}
    out = {}
    paras = doc["paragraphs"]
    for k, p in enumerate(paras):
        m = NAMED_RE.match(p["text"])
        # characterisation data follow the name, sometimes in the next paragraph (a page or column break)
        nxt = paras[k + 1]["text"] if k + 1 < len(paras) else ""
        if not m or not (CHARACTERISATION_RE.search(p["text"]) or CHARACTERISATION_RE.search(nxt[:400])):
            continue
        lab = norm_label(m.group(2))
        # a citation number glued to a word ("tert-Butyl589") and a small-caps gap ("L -phenyl") are not the name
        name = re.sub(r"(?<=[a-z])\d+(?=\s|$)", "", m.group(1))
        name = re.sub(r"(?<=[DLdl])\s+-", "-", name).strip()
        if not lab or UNIT_LABEL_RE.match(lab) or lab in out or len(re.findall(r"[A-Za-z]", name)) < 4:
            continue
        if name not in cache:
            try:
                cache[name] = helper._query_opsin_smiles(name)
            except Exception:
                cache[name] = None
        smi = cache[name]
        out[lab] = {"name": name, "smiles": smi if _mol(smi) is not None else None, "paragraph": p["id"]}
    if cache_path:
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def build_label_registry(figures, records, named=None, known_labels=None):
    """label -> {smiles, figure_id, figure, origin, seen_in, conflicts}. Origins in order of trust: "drawn" (the
    molecule agent saw the structure next to the label), "named" (the text names the compound, OPSIN gives the
    structure), "vision" (drawn in a figure that is not extracted, a spectrum, read by the vision models
    alone; kept only for labels the document uses elsewhere, known_labels, since the recogniser can read
    3o as 30), "reaction" (a labelled molecule of an extracted reaction). Within the best origin the structure
    seen in most figures wins, ties go to the earliest figure."""
    from rdkit.rdBase import BlockLogs
    cand = {}

    def add(label, smiles, source_id, origin):
        lab = norm_label(label)
        if not lab or not _concrete(smiles):
            return
        key = _canon(smiles)
        slot = cand.setdefault(lab, {}).setdefault(origin, {}).setdefault(key, {"smiles": smiles, "figures": []})
        if source_id not in slot["figures"]:
            slot["figures"].append(source_id)

    with BlockLogs():
        for fig in figures:
            rec = records.get(fig["figure_id"]) or {}
            for lab, smi in (rec.get("label_structures") or {}).items():
                add(lab, smi, fig["figure_id"], "drawn")
            for lab, smi in (rec.get("vision_labels") or {}).items():   # a skipped spectrum's compound
                if known_labels is None or norm_label(lab) in known_labels:
                    add(lab, smi, fig["figure_id"], "vision")
            for rx in _reactions(rec.get("output")):
                for side in ("reactants", "products", "conditions"):
                    for m in rx.get(side) or []:
                        if isinstance(m, dict) and m.get("smiles_source") != "label registry":
                            add(m.get("label"), m.get("smiles"), fig["figure_id"], "reaction")
        for lab, e in (named or {}).items():   # a paragraph id stands in for the figure
            if e.get("smiles"):
                add(lab, e["smiles"], e["paragraph"], "named")
        rank = {f["figure_id"]: i for i, f in enumerate(figures)}
        order = lambda x: rank.get(x, len(rank))   # noqa: E731  (text sources after every figure)
        names = {f["figure_id"]: f.get("name") for f in figures}
        registry = {}
        for lab, by_origin in cand.items():
            origin = next(o for o in ("drawn", "named", "vision", "reaction") if by_origin.get(o))
            options = by_origin[origin]
            best = max(options, key=lambda k: (len(options[k]["figures"]), -min(order(x) for x in options[k]["figures"])))
            first = min(options[best]["figures"], key=order)
            others = [{"smiles": v["smiles"], "figures": v["figures"], "origin": o}
                      for o, opts in by_origin.items() for k, v in opts.items()
                      if k != best and _skeleton(v["smiles"]) != _skeleton(options[best]["smiles"])]
            # a series label ("5" for every aryl iodide of a scope) names several compounds: one figure gives
            # it two different skeletons, or the document gives it three or more. It is listed, never used.
            per_fig, skels = {}, set()
            for opts in by_origin.values():
                for v in opts.values():
                    sk = _skeleton(v["smiles"])
                    skels.add(sk)
                    for fid in v["figures"]:
                        per_fig.setdefault(fid, set()).add(sk)
            ambiguous = len(skels) >= 3 or any(len(s) > 1 for s in per_fig.values())
            entry = {"label": lab, "smiles": options[best]["smiles"], "figure_id": first if first in rank else None,
                     "figure": names.get(first), "origin": origin, "ambiguous": ambiguous,
                     "seen_in": sorted({x for o in by_origin.values() for v in o.values() for x in v["figures"] if x in rank},
                                       key=order),
                     "conflicts": others}
            if named and lab in named:
                entry["name"] = named[lab]["name"]
                entry["named_in"] = named[lab]["paragraph"]
            registry[lab] = entry
    return registry


def resolve_labels(fig, rec, registry):
    """Molecules of one figure's reactions take the registry structure of their label (in place) when they have
    none, only a template with placeholders, or a structure this figure did not draw (built from a template
    row) that differs from the one drawn elsewhere. A structure the figure draws itself is kept; a different
    registry structure is then recorded under label_conflict. Salt and free-base forms count as the same."""
    from rdkit.rdBase import BlockLogs
    from chemietoolkit import helper
    stats = {"filled": 0, "replaced_template": 0, "replaced_built": 0, "conflicts": 0, "conditions": 0}
    usable = {lab: e for lab, e in registry.items() if not e.get("ambiguous")}
    own_labels = {norm_label(lab) for lab in (rec.get("label_structures") or {})}

    def mark(m, entry):
        m["smiles_source"] = "label registry"
        src = {"figure_id": entry["figure_id"], "figure": entry["figure"], "origin": entry["origin"]}
        if entry["origin"] == "named":
            src.update(name=entry.get("name"), paragraph=entry.get("named_in"))
        m["label_structure_from"] = src

    with BlockLogs():
        for rx in _reactions(rec.get("output")):
            for side in ("reactants", "products"):
                for m in rx.get(side) or []:
                    if not isinstance(m, dict):
                        continue
                    entry = usable.get(norm_label(m.get("label")) or "")
                    if not entry:
                        continue
                    cur = m.get("smiles")
                    if _mol(cur) is None:
                        if cur not in (None, "", "None"):
                            m["smiles_unreadable"] = cur
                        m["smiles"] = entry["smiles"]
                        mark(m, entry)
                        stats["filled"] += 1
                    elif "*" in str(cur):
                        m["smiles_template"] = cur
                        m["smiles"] = entry["smiles"]
                        mark(m, entry)
                        stats["replaced_template"] += 1
                    elif _skeleton(cur) != _skeleton(entry["smiles"]):
                        drawn_here = norm_label(m.get("label")) in own_labels
                        if entry["origin"] in ("drawn", "named") and entry["figure_id"] != fig["figure_id"] and not drawn_here:
                            # this figure does not draw the label, so its structure was built (a template with
                            # the row's substituents, a guess); a drawing elsewhere or the compound's name wins
                            m["smiles_before_label"] = cur
                            m["smiles"] = entry["smiles"]
                            mark(m, entry)
                            stats["replaced_built"] += 1
                        else:
                            m["label_conflict"] = {"registry_smiles": entry["smiles"], "figure_id": entry["figure_id"],
                                                   "figure": entry["figure"]}
                            stats["conflicts"] += 1
            for item in rx.get("conditions") or []:
                if not isinstance(item, dict) or helper._condition_role(item) not in helper._CHEMICAL_CONDITION_ROLES:
                    continue
                # only an entry that IS the label ("B17", label field "5a") takes its structure; a label
                # number inside free text ("NaOH (5 equiv)", "5 equiv of 2") is an amount, not a compound
                keys = [norm_label(item.get("label")), norm_label(_name_only(item.get("text")))]
                entry = next((usable[k] for k in keys if k and k in usable), None)
                if entry is None:
                    continue
                cur = item.get("smiles")
                m = _mol(cur)
                if m is not None and (m.GetNumHeavyAtoms() > helper._DEGENERATE_LOOKUP_ATOMS
                                      or _skeleton(cur) == _skeleton(entry["smiles"])):
                    continue
                item["smiles"] = entry["smiles"]
                mark(item, entry)
                item.pop("smiles_unresolved", None)
                stats["conditions"] += 1
    return stats


# ================================================================== text context
TEXT_CONTEXT_PROMPT = """You add information that the article text states about reactions already extracted from one figure of the same article. You never invent, infer or use chemical knowledge.

Figure: {name} (page {page})

Text sources, each with an id:
{sources}

Reactions extracted from this figure (JSON):
{reactions}

Tasks:
1. "figure_conditions": conditions that the caption or a footnote states for all reactions of this figure (for example "Reaction conditions: 1a (0.2 mmol), 2a (0.3 mmol), CuI (10 mol %), DMF (2 mL), 80 °C, 12 h"). Give one item per reagent, catalyst, solvent, temperature, time or atmosphere. Leave out the substrates and products themselves.
2. "reactions": for each reaction id, the items the text states for that specific reaction (the text names its product or reactant label, or its table entry) that the reaction does not already contain:
   - "conditions": role reagent, catalyst, solvent, additive, ligand, base, temperature, time, atmosphere, light or other;
   - "additional_info": role yield, ee, er, dr, conversion, selectivity or note.

Rules:
- Every item has "role", "text", "evidence" and "source". "evidence" is an exact, verbatim quote (5 to 40 words) copied from the source named by "source". Items whose evidence is not literally in that source are discarded.
- "text" is the value as printed (for example "CuI (10 mol %)", "80 °C", "92% ee"), and it must appear inside the evidence.
- Skip anything the reaction already contains (the same reagent, temperature or yield, even if written differently).
- Do not write SMILES. If nothing applies, return empty lists.

Respond with one JSON object and nothing else:
{{"figure_conditions": [{{"role": "solvent", "text": "DMF", "evidence": "...", "source": "fn_a"}}], "reactions": [{{"reaction_id": "...", "conditions": [], "additional_info": []}}]}}
"""


def text_sources(fig, doc, rec, max_paragraphs=8, max_chars=9000):
    """The caption, footnotes, printed box text and related paragraphs of one figure, as (id, text)."""
    out = []
    if fig.get("caption"):
        out.append(("caption", fig["caption"]))
    for fn in fig.get("footnotes") or []:
        out.append((fn["id"], "%s %s" % (fn["marker"], fn["text"])))
    if fig.get("box_text"):
        out.append(("box", fig["box_text"][:3000]))
    labels = set()
    for rx in _reactions(rec.get("output")):
        for side in ("reactants", "products"):
            for m in rx.get(side) or []:
                lab = norm_label(m.get("label")) if isinstance(m, dict) else None
                if lab:
                    labels.add(lab)
    scored = []
    for k, p in enumerate(doc["paragraphs"]):
        cites = p["id"] in fig.get("cited_by", [])
        hits = len(labels & set(p.get("labels") or []))
        if cites or hits:
            scored.append((-(10 * cites + hits), k, p))
    used = sum(len(t) for _, t in out)
    picked = []
    for _, _, p in sorted(scored)[:max_paragraphs]:
        if used + len(p["text"]) > max_chars:
            continue
        picked.append(p)
        used += len(p["text"])
    for p in sorted(picked, key=lambda p: doc["paragraphs"].index(p)):
        out.append((p["id"], p["text"]))
    return out


def _compact_reaction(rx, k):
    def mol(m):
        d = {"label": m.get("label")} if str(m.get("label") or "").strip() not in ("", "None") else {}
        d["smiles"] = m.get("smiles")
        return d
    return {"reaction_id": str(rx.get("reaction_id", k)),
            "reactants": [mol(m) for m in rx.get("reactants") or [] if isinstance(m, dict)],
            "products": [mol(m) for m in rx.get("products") or [] if isinstance(m, dict)],
            "conditions": [{x: c.get(x) for x in ("role", "text") if c.get(x)} for c in rx.get("conditions") or [] if isinstance(c, dict)],
            "additional_info": [{x: c.get(x) for x in ("role", "text") if c.get(x)} for c in rx.get("additional_info") or [] if isinstance(c, dict)]}


def _valid_item(item, sources, roles):
    if not isinstance(item, dict):
        return None, "not an object"
    role = str(item.get("role") or "").strip().lower()
    text, ev, src = str(item.get("text") or "").strip(), str(item.get("evidence") or "").strip(), str(item.get("source") or "").strip()
    if role not in roles:
        return None, "role %r" % role
    if not text or not ev or src not in sources:
        return None, "missing text, evidence or source"
    if norm_text(ev) not in norm_text(sources[src]):
        return None, "evidence not in source"
    if norm_text(text) not in norm_text(ev):
        return None, "text not in evidence"
    return {"role": role, "text": text, "source": "text", "evidence": {"source_id": src, "quote": ev}}, None


GENERIC_NAMES = {"ligand", "base", "acid", "catalyst", "solvent", "additive", "reagent", "oxidant", "reductant",
                 "promoter", "initiator", "photocatalyst", "substrate", "reactant", "product", "starting material"}


AMOUNT_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:equiv\.?|eq\.?|mol\s*%|mmol|mol|mg|g|kg|ml|μl|ul|l|mm|m)(?![A-Za-z])\.?"
                       r"\s*(?:of\s+)?", re.I)
MASS_RE = re.compile(r"\d\s*(?:mg|g|kg)(?![A-Za-z])")


AMOUNT_IN_BRACKETS_RE = re.compile(r"\d\s*(?:equiv|eq(?![a-z])|mol|mmol|mg|g(?![a-z])|kg|ml|μl|ul|l(?![a-z])|m(?![a-z])"
                                   r"|%|°|h(?![a-z])|min)", re.I)


def _name_only(text):
    """'triethylamine (3 equiv)' or '20 mol % of TfOH' -> the substance name alone. Only brackets holding an
    amount go; formula brackets stay (Pd(OAc)2, Pd(PPh3)4)."""
    s = re.sub(r"\s*\(([^()]*)\)", lambda m: "" if AMOUNT_IN_BRACKETS_RE.search(m.group(1)) else m.group(0),
               str(text or ""))
    return AMOUNT_RE.sub("", s).strip(" ,;:")


def _already_has(rx, item):
    """The reaction already names this item (same text) or already holds its structure (same skeleton,
    among conditions, reactants or products)."""
    t = norm_text(item["text"])
    for key in ("conditions", "additional_info"):
        for c in rx.get(key) or []:
            if isinstance(c, dict):
                have = norm_text(c.get("text") or c.get("smiles") or "")
                if have and (t in have or have in t):
                    return True
    skel = _skeleton(item.get("smiles"))
    if skel:
        for key in ("conditions", "reactants", "products"):
            for c in rx.get(key) or []:
                if isinstance(c, dict) and _skeleton(c.get("smiles")) == skel:
                    return True
    return False


def add_text_context(fig, doc, rec, client, model_name, mk):
    """One text-only call per figure; verified items are added to its reactions (in place)."""
    import llm_client as llm
    from chemietoolkit import helper
    rxns = _reactions(rec.get("output"))
    sources = text_sources(fig, doc, rec)
    report = {"sources": [s for s, _ in sources], "added": 0, "dropped": [], "model": model_name}
    if not rxns or not sources:
        return report
    src_map = dict(sources)
    prompt = TEXT_CONTEXT_PROMPT.format(
        name=fig.get("name") or fig["figure_id"], page=fig["page"],
        sources="\n\n".join("[%s] %s" % (sid, txt) for sid, txt in sources),
        reactions=json.dumps([_compact_reaction(r, k) for k, r in enumerate(rxns)], ensure_ascii=False))
    response, _ = llm.final_json_call(
        client, model_name,
        [{"role": "system", "content": "You attach text evidence to extracted chemical reactions."},
         {"role": "user", "content": prompt}], mk, stream=True)
    reply = llm.loads_lenient(response.choices[0].message.content)
    report["raw"] = response.choices[0].message.content
    by_id = {str(rx.get("reaction_id", k)): rx for k, rx in enumerate(rxns)}
    structures = {}
    distinct = set()

    def checked(raw, roles):
        """The verified item with its structure (names resolved the way the pipeline resolves conditions),
        or None with the reason recorded."""
        item, why = _valid_item(raw, src_map, roles)
        if item is None:
            report["dropped"].append({"item": raw, "reason": why})
            return None
        if item["role"] == "yield" and "%" not in item["text"] and MASS_RE.search(item["text"]):
            report["dropped"].append({"item": raw, "reason": "a mass, not a yield"})
            return None
        if item["role"] in CHEMICAL_ROLES:
            name = _name_only(item["text"])
            if len(re.findall(r"[A-Za-z]", name)) < 2 or name.lower() in GENERIC_NAMES:
                report["dropped"].append({"item": raw, "reason": "no substance named (an amount or a generic word)"})
                return None
            if name not in structures:
                probe = {"role": item["role"], "text": name}
                try:
                    helper.fallback_resolve_condition_smiles_in_data({"conditions": [probe]})
                except Exception:
                    pass
                structures[name] = probe.get("smiles") if _mol(probe.get("smiles")) is not None else None
            if structures[name]:
                item["smiles"] = structures[name]
        return item

    def add(rx, item, key):
        if _already_has(rx, item):
            return
        rx.setdefault(key, []).append(copy.deepcopy(item))
        report["added"] += 1
        distinct.add((key, item["role"], item["text"], item.get("applies_to")))

    if isinstance(reply, dict):
        for raw in reply.get("figure_conditions") or []:
            item = checked(raw, CONDITION_ROLES)
            if item is not None:
                item["applies_to"] = "all reactions of the figure"
                for rx in rxns:
                    add(rx, item, "conditions")
        for entry in reply.get("reactions") or []:
            rx = by_id.get(str((entry or {}).get("reaction_id")))
            if rx is None:
                continue
            for raw in entry.get("conditions") or []:
                item = checked(raw, CONDITION_ROLES)
                if item is not None:
                    add(rx, item, "conditions")
            for raw in entry.get("additional_info") or []:
                item = checked(raw, RESULT_ROLES)
                if item is not None:
                    add(rx, item, "additional_info")
    report["distinct_added"] = len(distinct)
    return report


# ================================================================== main text agent: figure roles
FIGURE_ROLES = ("reaction_scheme", "substrate_scope", "optimization", "mechanism", "control_experiment",
                "synthetic_application", "scale_up", "compound_structures", "spectra_or_data", "toc_graphic", "other")
REACTIONS_ARE = ("performed", "proposed", "none")
ROLE_RULES = ((r"mechanis|catalytic cycle", "mechanism", "proposed"),
              (r"control experiment|mechanistic (?:study|studies|experiment|investigation)", "control_experiment", "performed"),
              (r"optimi[sz]ation|screening of", "optimization", "performed"),
              (r"gram[- ]scale|scale[- ]up|large[- ]scale", "scale_up", "performed"),
              (r"scope", "substrate_scope", "performed"),
              (r"application|transformation|derivati[sz]ation|total synthesis", "synthetic_application", "performed"),
              (r"\bnmr\b|spectr|chromatogram|hplc|x-ray|crystal structure|voltammogram|uv[- ]vis", "spectra_or_data", "none"),
              (r"bioactive|natural products|representative (?:compounds|molecules|drugs)|structures of", "compound_structures", "none"))
SPECTRA_BOX_RE = re.compile(r"\bppm\b|\bf1\b|chemical shift|\bm/z\b|absorbance|wavelength|retention time|intensity", re.I)
REACTION_WORDS_RE = re.compile(r"reaction|synthes|scope|condition|yield|preparation|cycli|coupling", re.I)

FIGURE_ROLE_PROMPT = """You label what each figure of a chemistry article shows, from its caption, footnotes, printed text and the sentences that cite it. Do not guess from chemical knowledge.

Roles:
- reaction_scheme: a reaction or a synthetic route
- substrate_scope: one reaction applied to many substrates (products with yields)
- optimization: the same reaction under different conditions (entries)
- mechanism: a proposed or plausible mechanism, catalytic cycle or intermediates
- control_experiment: control or mechanistic experiments that were run
- synthetic_application: products used further (derivatization, transformations, total synthesis)
- scale_up: gram-scale or large-scale runs
- compound_structures: structures only (bioactive compounds, catalysts, ligands), no reaction
- spectra_or_data: spectra, chromatograms, plots, crystal structures, data tables that are not reactions
- toc_graphic: the graphical abstract
- other

For every figure give "role", "reactions_are" ("performed": reactions that were run; "proposed": hypothetical steps such as a mechanism; "none"), "evidence": a verbatim quote (3 to 30 words) from that figure's texts below, and "source": the id of the text the quote comes from.

Figures:
{figures}

Respond with one JSON object and nothing else:
{{"figures": [{{"figure_id": "main_p2_1", "role": "substrate_scope", "reactions_are": "performed", "evidence": "...", "source": "caption"}}]}}
"""


def _sentences_citing(text, name):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if name and name in s][:2]


def figure_texts(fig, doc):
    """The texts a figure's role is read from: caption, footnotes, printed box text, sentences citing it."""
    out = []
    if fig.get("caption"):
        out.append(("caption", fig["caption"][:600]))
    for fn in (fig.get("footnotes") or [])[:3]:
        out.append((fn["id"], ("%s %s" % (fn["marker"], fn["text"]))[:300]))
    if fig.get("box_text"):
        out.append(("box", re.sub(r"\s+", " ", fig["box_text"])[:300]))
    by_id = {p["id"]: p for p in doc["paragraphs"]}
    for pid in (fig.get("cited_by") or [])[:3]:
        for s in _sentences_citing(by_id[pid]["text"], fig.get("name")):
            out.append((pid, s[:300]))
    return out


def _role_by_rule(fig):
    caption = re.sub(CAPTION_RE, "", fig.get("caption") or "").lower()
    for pattern, role, are in ROLE_RULES:
        if caption and re.search(pattern, caption):
            return {"role": role, "reactions_are": are, "by": "caption rule", "verified": True,
                    "evidence": {"source_id": "caption", "quote": fig["caption"][:120]}}
    if not fig.get("caption") and SPECTRA_BOX_RE.search(fig.get("box_text") or ""):
        return {"role": "spectra_or_data", "reactions_are": "none", "by": "box text rule", "verified": True,
                "evidence": {"source_id": "box", "quote": (fig.get("box_text") or "")[:120]}}
    return None


def classify_figures(doc, client, model_name, mk, batch=25, image_fallback=True):
    """fig["role"] for every figure: the text-based label first (verbatim evidence required), else the
    caption or box-text rule, else the unverified label. One call per `batch` figures with text; a figure
    with no text at all is read from its image (one call each, image_fallback)."""
    import llm_client as llm
    reports = []
    with_text = [f for f in doc["figures"] if figure_texts(f, doc)]
    for start in range(0, len(with_text), batch):
        chunk = with_text[start:start + batch]
        texts = {f["figure_id"]: dict(figure_texts(f, doc)) for f in chunk}
        listing = "\n\n".join("%s (%s):\n%s" % (f["figure_id"], f.get("name") or "unnamed",
                                                 "\n".join("  [%s] %s" % kv for kv in texts[f["figure_id"]].items()))
                              for f in chunk)
        reply = {}
        try:
            response, _ = llm.final_json_call(
                client, model_name,
                [{"role": "system", "content": "You classify the figures of chemistry articles."},
                 {"role": "user", "content": FIGURE_ROLE_PROMPT.format(figures=listing)}], mk, stream=True)
            reply = llm.loads_lenient(response.choices[0].message.content) or {}
        except Exception as exc:
            reports.append("figure roles call failed: %r" % exc)
        answers = {str(x.get("figure_id")): x for x in (reply.get("figures") or []) if isinstance(x, dict)} if isinstance(reply, dict) else {}
        for f in chunk:
            a = answers.get(f["figure_id"])
            if a and a.get("role") in FIGURE_ROLES:
                src = str(a.get("source") or "")
                ev = str(a.get("evidence") or "")
                verified = bool(ev) and src in texts[f["figure_id"]] and norm_text(ev) in norm_text(texts[f["figure_id"]][src])
                are = a.get("reactions_are") if a.get("reactions_are") in REACTIONS_ARE else None
                f["role"] = {"role": a["role"], "reactions_are": are, "by": "text", "verified": verified,
                             "evidence": {"source_id": src, "quote": ev}}
    for f in doc["figures"]:
        rule = _role_by_rule(f)
        current = f.get("role")
        if current is None or not current.get("verified"):
            f["role"] = rule or current or {"role": "other", "reactions_are": None, "by": "no text", "verified": False}
        if image_fallback and f["role"]["by"] == "no text":
            try:
                f["role"] = classify_by_image(f, client, model_name, mk) or f["role"]
            except Exception as exc:
                reports.append("image role of %s failed: %r" % (f["figure_id"], exc))
    return reports


IMAGE_ROLE_PROMPT = """The image is one region cut from a page of a chemistry article or its supporting information. It has no caption or printed text to go by. Say what it shows.

Roles: reaction_scheme, substrate_scope, optimization, mechanism, control_experiment, synthetic_application, scale_up, compound_structures (structures only, no reaction), spectra_or_data (NMR, MS or IR spectra, chromatograms, plots, crystal structures, even when a structure is drawn beside them), toc_graphic, other.

Respond with one JSON object and nothing else:
{"role": "spectra_or_data", "reactions_are": "none", "description": "1H and 13C NMR spectra of 3b"}
"""


def classify_by_image(fig, client, model_name, mk):
    """The role of a figure without any text, read from its image (one call); marked "by": "image"."""
    import base64
    import llm_client as llm
    with open(fig["image"], "rb") as f:
        img = base64.b64encode(f.read()).decode()
    response, _ = llm.final_json_call(
        client, model_name,
        [{"role": "system", "content": "You classify the figures of chemistry articles."},
         {"role": "user", "content": [{"type": "text", "text": IMAGE_ROLE_PROMPT},
                                      {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}],
        mk, stream=True)
    reply = llm.loads_lenient(response.choices[0].message.content) or {}
    if not isinstance(reply, dict) or reply.get("role") not in FIGURE_ROLES:
        return None
    are = reply.get("reactions_are") if reply.get("reactions_are") in REACTIONS_ARE else None
    return {"role": reply["role"], "reactions_are": are, "by": "image", "verified": False,
            "description": str(reply.get("description") or "")[:200]}


def skip_extraction(fig):
    """Spectra, chromatograms and data plots hold no reaction: no main.ChemEagle run for them. A text label
    needs its evidence; an image label or the box-text rule (axis text such as 'ppm') is enough on its own."""
    role = fig.get("role") or {}
    if role.get("role") != "spectra_or_data":
        return False
    if role.get("by") in ("box text rule", "image"):
        return True
    return bool(role.get("verified")) and not REACTION_WORDS_RE.search(fig.get("caption") or "")


def harvest_drawn_labels(fig, out_dir):
    """label -> SMILES drawn in a figure that is not extracted (a spectrum with its compound beside it), read by
    the local vision models alone (MolDetector, Image2Graph, OCR; no LLM). Cached in out_dir/figures."""
    cache = os.path.join(out_dir, "figures", fig["figure_id"] + "_vision_labels.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    import get_molecular_agent
    from chemietoolkit import helper
    try:
        labels = helper.drawn_labels_from_mol_result(get_molecular_agent.extract_molecule_corefs(fig["image"]))
    except Exception as exc:
        print("[pdf] WARNING vision labels of %s failed: %r" % (fig["figure_id"], exc), flush=True)
        return {}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(labels, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return labels


# ================================================================== main text agent: reactions in the text
TEXT_REACTION_PROMPT = """You read part of the text of a chemistry article and list the chemical reactions it says were carried out. You never invent, infer or use chemical knowledge.

Compounds of this article (label: name when the text names it):
{compounds}

Reactions already extracted from the article's figures (product <- reactants, by label):
{figure_reactions}

Text (each paragraph starts with its id in brackets):
{text}

List every reaction the text states was performed in this work: a procedure (general or for one compound), the preparation of a starting material, a transformation of a product, a control experiment. Skip reactions that are only cited from other work or only proposed (mechanisms).
For each reaction give:
- "reactants" and "products": lists of {{"label": the printed compound label or "", "name": the compound name as printed or ""}}, each with a label or a name;
- "conditions": [{{"role": "reagent|catalyst|solvent|temperature|time|atmosphere|other", "text": "..."}}], as printed;
- "yield": as printed, or "";
- "evidence": an exact, verbatim quote (5 to 60 words) from the paragraph that states the reaction, and "source": that paragraph id;
- "in_figure": true when it is one of the figure reactions listed above.
Never write SMILES. Copy labels and names exactly as printed. If there is none, return an empty list.

Respond with one JSON object and nothing else:
{{"reactions": [{{"reactants": [{{"label": "1a", "name": ""}}], "products": [{{"label": "3a", "name": ""}}], "conditions": [], "yield": "85%", "evidence": "...", "source": "main:p12", "in_figure": false}}]}}
"""
DATA_START_RE = re.compile(r"\b(?:1H|13C|19F|31P)\s*NMR\b|\bHRMS\b|\bIR\s*\(|\bRf\s*=")


def _text_chunks(doc, max_chars=6000):
    """Paragraphs of the article and SI (references and front matter left out), characterisation data cut,
    grouped into chunks of about max_chars."""
    chunks, cur, size = [], [], 0
    for p in doc["paragraphs"]:
        if any(x in p["section"].lower() for x in NO_LINK_SECTIONS):
            continue
        text = p["text"]
        m = DATA_START_RE.search(text)
        if m and m.start() > 60:
            text = text[:m.start()].rstrip()
        if len(text) < 40:
            continue
        if cur and size + len(text) > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append((p["id"], text))
        size += len(text)
    if cur:
        chunks.append(cur)
    return chunks


def _label_in(label, text):
    lab = norm_label(label)
    return bool(lab) and any(lab in expand_label_token(t) for t in re.findall(r"[\w'′‐-―−,]+", text))


def extract_text_reactions(doc, registry, named, figure_reactions, client, model_name, mk, out_dir=None):
    """Reactions the text states were run. Each is kept only with its quote found verbatim in the cited
    paragraph and with every compound's label or name printed there; structures come from the label table
    or OPSIN, never from the LLM. A reaction a figure already holds becomes a text_mention of it."""
    import llm_client as llm
    from chemietoolkit import helper
    usable = {lab: e for lab, e in registry.items() if not e.get("ambiguous")}
    comp_lines = []
    for lab, e in sorted(usable.items(), key=lambda kv: (len(kv[0]), kv[0]))[:200]:
        comp_lines.append("%s: %s" % (lab, e.get("name") or (named.get(lab) or {}).get("name") or "(drawn)"))
    fig_lines = []
    for rx in figure_reactions[:150]:
        prods = [m.get("label") for m in rx.get("products") or [] if isinstance(m, dict) and m.get("label")]
        reacts = [m.get("label") for m in rx.get("reactants") or [] if isinstance(m, dict) and m.get("label")]
        if prods:
            fig_lines.append("%s: %s <- %s" % (rx.get("figure") or rx.get("figure_id"), ", ".join(map(str, prods)), " + ".join(map(str, reacts)) or "?"))
    cache_path = os.path.join(out_dir, "named_compounds.json") if out_dir else None
    names_cache = json.load(open(cache_path, encoding="utf-8")) if cache_path and os.path.exists(cache_path) else {}
    para_text = {p["id"]: p["text"] for p in doc["paragraphs"]}
    para_of = {p["id"]: p for p in doc["paragraphs"]}
    report = {"calls": 0, "proposed": 0, "kept": 0, "dropped": []}
    found = []

    def structure(mol):
        lab = norm_label(mol.get("label"))
        if lab and lab in usable:
            return usable[lab]["smiles"], "label registry (%s)" % usable[lab]["origin"]
        name = str(mol.get("name") or "").strip()
        if len(re.findall(r"[A-Za-z]", name)) >= 4:
            if name not in names_cache:
                try:
                    names_cache[name] = helper._query_opsin_smiles(name)
                except Exception:
                    names_cache[name] = None
            if _mol(names_cache[name]) is not None:
                return names_cache[name], "name (OPSIN)"
        return None, None

    for chunk in _text_chunks(doc):
        report["calls"] += 1
        prompt = TEXT_REACTION_PROMPT.format(
            compounds="\n".join(comp_lines) or "(none)", figure_reactions="\n".join(fig_lines) or "(none)",
            text="\n\n".join("[%s] %s" % (pid, t) for pid, t in chunk))
        try:
            response, _ = llm.final_json_call(
                client, model_name,
                [{"role": "system", "content": "You extract the reactions stated in chemistry article text."},
                 {"role": "user", "content": prompt}], mk, stream=True)
            reply = llm.loads_lenient(response.choices[0].message.content) or {}
        except Exception as exc:
            report["dropped"].append({"reason": "call failed: %r" % exc})
            continue
        ids = {pid for pid, _ in chunk}
        for raw in (reply.get("reactions") or []) if isinstance(reply, dict) else []:
            if not isinstance(raw, dict):
                continue
            report["proposed"] += 1
            src, ev = str(raw.get("source") or ""), str(raw.get("evidence") or "")
            if src not in ids or not ev or norm_text(ev) not in norm_text(para_text[src]):
                report["dropped"].append({"reaction": raw, "reason": "evidence not in the cited paragraph"})
                continue
            text = para_text[src]
            sides = {}
            for side in ("reactants", "products"):
                mols = []
                for m in raw.get(side) or []:
                    if not isinstance(m, dict):
                        continue
                    lab, name = str(m.get("label") or "").strip(), str(m.get("name") or "").strip()
                    lab_ok = bool(lab) and _label_in(lab, text)
                    name_ok = bool(name) and norm_text(name) in norm_text(text)
                    if not (lab_ok or name_ok):
                        continue          # a compound the paragraph does not print
                    mol = {"label": lab if lab_ok else "", "name": name if name_ok else ""}
                    smi, how = structure(mol)
                    mol["smiles"] = smi
                    if how:
                        mol["smiles_source"] = how
                    else:
                        mol["smiles_unresolved"] = True
                    mols.append({k: v for k, v in mol.items() if v not in ("", None)} | {"smiles": smi})
                sides[side] = mols
            if not sides.get("products"):
                report["dropped"].append({"reaction": raw, "reason": "no product printed in the paragraph"})
                continue
            if not sides.get("reactants"):   # "4e was synthesized according to ref 40" states a product, not a reaction
                report["dropped"].append({"reaction": raw, "reason": "no reactant printed in the paragraph"})
                continue
            conds = [{"role": str(c.get("role") or "other").lower(), "text": str(c.get("text")).strip(), "source": "text"}
                     for c in raw.get("conditions") or [] if isinstance(c, dict) and c.get("text")
                     and norm_text(c.get("text")) in norm_text(text)]
            chem = [c for c in conds if c["role"] in CHEMICAL_ROLES | {"reagent"}]
            for c in chem:
                name = _name_only(c["text"])
                if len(re.findall(r"[A-Za-z]", name)) >= 2 and name.lower() not in GENERIC_NAMES:
                    probe = {"role": c["role"], "text": name}
                    try:
                        helper.fallback_resolve_condition_smiles_in_data({"conditions": [probe]})
                    except Exception:
                        pass
                    if _mol(probe.get("smiles")) is not None:
                        c["smiles"] = probe["smiles"]
            info = []
            y = str(raw.get("yield") or "").strip()
            if y and norm_text(y) in norm_text(text):
                info.append({"role": "yield", "text": y, "source": "text"})
            found.append({"reactants": sides.get("reactants") or [], "products": sides["products"],
                          "conditions": conds, "additional_info": info,
                          "evidence": {"source_id": src, "quote": ev}, "llm_in_figure": bool(raw.get("in_figure")),
                          "doc": para_of[src]["doc"], "page": para_of[src]["pages"][0], "section": para_of[src]["section"]})
    if cache_path:
        json.dump(names_cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def product_key(rx):
        skels = sorted(filter(None, (_skeleton(m.get("smiles")) for m in rx.get("products") or [] if isinstance(m, dict))))
        if skels:
            return ("skel",) + tuple(skels)
        labs = sorted(filter(None, (norm_label(m.get("label")) for m in rx.get("products") or [] if isinstance(m, dict))))
        return ("label",) + tuple(labs) if labs else None

    by_key = {}
    for rx in figure_reactions:
        k = product_key(rx)
        if k:
            by_key.setdefault(k, rx)
            labs = sorted(filter(None, (norm_label(m.get("label")) for m in rx.get("products") or [] if isinstance(m, dict))))
            if labs:
                by_key.setdefault(("label",) + tuple(labs), rx)
    new, mentions = [], 0
    for t in found:
        keys = [product_key(t)]
        labs = sorted(filter(None, (norm_label(m.get("label")) for m in t["products"])))
        if labs:
            keys.append(("label",) + tuple(labs))
        hit = next((by_key[k] for k in keys if k and k in by_key), None)
        mention = {"source_id": t["evidence"]["source_id"], "quote": t["evidence"]["quote"]}
        if t["additional_info"]:
            mention["yield"] = t["additional_info"][0]["text"]
        if hit is not None:
            if mention not in hit.setdefault("text_mentions", []):
                hit["text_mentions"].append(mention)
                mentions += 1
            continue
        if not any(m.get("label") or m.get("smiles") for m in t["products"]):
            t["general_procedure"] = True   # "the desired product" of "a phenol": a procedure, no compound named
        t["reaction_id"] = "t%d" % (len(new) + 1)
        t["doc_reaction_id"] = "text:" + t["reaction_id"]
        t["source"] = "text"
        t["figure_id"] = t["figure"] = None
        for k in keys:
            if k:
                by_key[k] = t
        new.append(t)
    report.update(kept=len(new), mentions=mentions)
    return new, report


# ================================================================== the whole document
def ChemEagle_PDF(
    pdf_path: str,
    *,
    si_paths: Optional[Sequence[str]] = None,
    out_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
    use_network: Optional[bool] = None,
    use_confidence: Optional[bool] = None,
    confidence_llm: Optional[bool] = None,
    confidence_checker: Optional[str] = None,
    use_label_registry: bool = True,
    use_text_names: bool = True,
    use_text_context: bool = True,
    use_figure_roles: bool = True,
    skip_spectra: bool = True,
    use_text_reactions: bool = True,
    text_model: Optional[str] = None,
    dpi: int = 200,
    large_detector: bool = True,
    resume: bool = True,
) -> dict:
    """Extract every reaction of an article PDF (plus its SI PDFs) as one document.

    Args:
        pdf_path: the article PDF. si_paths: SI PDFs read into the same document (labels and figure
            references cross between them).
        out_dir: where crops, per-figure results, logs and document.json go (default: <pdf name>_chemeagle
            next to the PDF). Per-figure results are reused on a rerun unless resume=False.
        model_name, base_url, api_key, use_plan_observer, use_action_observer, use_network, use_confidence,
            confidence_llm, confidence_checker: passed to main.ChemEagle for every figure.
        use_label_registry: fill molecules that only name a compound label with the structure drawn for that
            label elsewhere in the document (no API cost).
        use_text_names: also take label structures from the compound names of characterisation paragraphs
            ("tert-Butyl Benzoyl-L-phenylalaninate (3a). ...") through OPSIN; ranked below drawn structures.
        use_text_context: one text-only LLM call per figure that adds conditions, yields and selectivities
            the text states, each with verbatim evidence.
        use_figure_roles: label each figure (substrate scope, optimization, mechanism, control experiment,
            spectra ...) from its caption and citing sentences; reactions keep their figure's role, and those
            of a mechanism are marked proposed_step. skip_spectra: no extraction run for figures labelled
            spectra or data plots (they hold no reaction).
        use_text_reactions: read the article and SI text for reactions it states were run; one already in a
            figure gets the passage as a text_mention, a new one is added with "source": "text".
        text_model: the model of every text-only call (figure roles, text context, text reactions); default
            the extraction model.
        dpi, large_detector: page rendering resolution and VisualHeist size for the figure crops.

    Returns the document: {document, figures, paragraphs, compounds, reactions, summary}, also written to
    out_dir/document.json.
    """
    pdf_path = os.path.abspath(pdf_path)
    si_paths = [os.path.abspath(p) for p in (si_paths or [])]
    out_dir = os.path.abspath(out_dir or os.path.splitext(pdf_path)[0] + "_chemeagle")
    os.makedirs(out_dir, exist_ok=True)
    files = [("main", pdf_path)] + [("si%d" % (i + 1) if len(si_paths) > 1 else "si", p) for i, p in enumerate(si_paths)]
    chemeagle_kwargs = dict(model_name=model_name, base_url=base_url, api_key=api_key,
                            use_plan_observer=use_plan_observer, use_action_observer=use_action_observer,
                            use_network=use_network, use_confidence=use_confidence,
                            confidence_llm=confidence_llm, confidence_checker=confidence_checker)
    t0 = time.time()
    notes = []
    with _in_repo():
        import llm_client as llm
        print("[pdf] reading %d file(s): pages, text layer, figure boxes" % len(files), flush=True)
        doc = parse_document(files, out_dir, dpi=dpi, large_detector=large_detector)
        print("[pdf] %d figures, %d paragraphs" % (len(doc["figures"]), len(doc["paragraphs"])), flush=True)

        client = tm = mk = None
        if use_figure_roles or use_text_context or use_text_reactions:
            tm = text_model or llm.resolve_model(model_name)
            client = llm.get_client(api_key=llm.resolve_key(api_key), base_url=llm.resolve_base_url(base_url))
            mk = llm.model_kwargs(tm)
        if use_figure_roles:
            notes += classify_figures(doc, client, tm, mk)
            print("[pdf] figure roles: %s" % dict(_count(f["role"]["role"] for f in doc["figures"])), flush=True)

        records = {}
        for i, fig in enumerate(doc["figures"], start=1):
            label = "%s (%s, %s)" % (fig["figure_id"], fig.get("name") or "unnamed", (fig.get("role") or {}).get("role", "-"))
            if use_figure_roles and skip_spectra and skip_extraction(fig):
                records[fig["figure_id"]] = {"figure_id": fig["figure_id"], "status": "skipped",
                                             "reason": "spectra or data, no reaction", "output": {"reactions": []},
                                             "vision_labels": harvest_drawn_labels(fig, out_dir) if use_label_registry else {}}
                print("[pdf] figure %d/%d %s: skipped" % (i, len(doc["figures"]), label), flush=True)
                continue
            print("[pdf] figure %d/%d %s" % (i, len(doc["figures"]), label), flush=True)
            records[fig["figure_id"]] = run_figure(fig, out_dir, chemeagle_kwargs, resume=resume)
            rec = records[fig["figure_id"]]
            if rec.get("status") != "ok" and use_label_registry:
                # the extraction failed; the compounds the figure draws still reach the label table
                rec["vision_labels"] = harvest_drawn_labels(fig, out_dir)
            print("[pdf]   %s, %d reactions, %ss%s" % (rec["status"], len(_reactions(rec.get("output"))), rec.get("seconds"),
                                                 ": " + str(rec.get("error"))[:120] if rec.get("status") != "ok" else ""), flush=True)

        records = copy.deepcopy(records)   # the cached per-figure results stay as main.ChemEagle returned them
        named = named_compounds(doc, out_dir) if use_label_registry and use_text_names else {}
        known = {lab for p in doc["paragraphs"] for lab in p.get("labels") or []} | set(named)
        for rec in records.values():
            known |= {norm_label(lab) for lab in (rec.get("label_structures") or {})}
            for rx in _reactions(rec.get("output")):
                for side in ("reactants", "products", "conditions"):
                    known |= {norm_label(m.get("label")) for m in rx.get(side) or [] if isinstance(m, dict)}
        known.discard(None)
        registry = build_label_registry(doc["figures"], records, named, known) if use_label_registry else {}
        label_stats = {"filled": 0, "replaced_template": 0, "replaced_built": 0, "conflicts": 0, "conditions": 0}
        if use_label_registry:
            for fig in doc["figures"]:
                for k, v in resolve_labels(fig, records[fig["figure_id"]], registry).items():
                    label_stats[k] += v

        text_reports = {}
        if use_text_context:
            for fig in doc["figures"]:
                rec = records[fig["figure_id"]]
                if rec.get("status") != "ok" or not _reactions(rec.get("output")):
                    continue
                try:
                    text_reports[fig["figure_id"]] = add_text_context(fig, doc, rec, client, tm, mk)
                except Exception as exc:
                    text_reports[fig["figure_id"]] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
                    print("[pdf] WARNING text context of %s failed: %r" % (fig["figure_id"], exc), flush=True)

        # ---- figure reactions, with their figure's role
        reactions = []
        for fig in doc["figures"]:
            rec = records[fig["figure_id"]]
            out = rec.get("output") if isinstance(rec.get("output"), dict) else {}
            rxns = _reactions(out)
            fig.update(status=rec.get("status"), error=rec.get("error") or rec.get("reason"), agents=rec.get("agents"),
                       n_reactions=len(rxns), confidence_summary=out.get("confidence_summary"),
                       image=os.path.relpath(fig["image"], out_dir))
            rep = text_reports.get(fig["figure_id"])
            if rep is not None:
                fig["text_context"] = {k: rep.get(k) for k in ("sources", "distinct_added", "added", "dropped", "model", "error") if k in rep}
            else:
                fig["text_context"] = {"sources": [s for s, _ in text_sources(fig, doc, rec)]}
            role = fig.get("role") or {}
            for k, rx in enumerate(rxns):
                entry = dict(rx)
                entry["doc_reaction_id"] = "%s:%s" % (fig["figure_id"], rx.get("reaction_id", k))
                entry["figure_id"], entry["figure"], entry["doc"], entry["page"] = fig["figure_id"], fig.get("name"), fig["doc"], fig["page"]
                if role:
                    entry["figure_role"] = role.get("role")
                    if role.get("reactions_are") == "proposed" or role.get("role") == "mechanism":
                        entry["proposed_step"] = True
                reactions.append(entry)

        # ---- reactions stated in the text
        text_report = None
        if use_text_reactions:
            try:
                new_rxns, text_report = extract_text_reactions(doc, registry, named, reactions, client, tm, mk, out_dir)
                reactions += new_rxns
                print("[pdf] text reactions: %d new, %d mentions of figure reactions (%d calls)"
                      % (text_report["kept"], text_report["mentions"], text_report["calls"]), flush=True)
            except Exception as exc:
                text_report = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
                print("[pdf] WARNING text reactions failed: %r" % exc, flush=True)

    # ---- assemble
    for lab, e in registry.items():
        e["mentioned_in"] = [p["id"] for p in doc["paragraphs"] if lab in (p.get("labels") or [])]
    compounds = sorted(registry.values(), key=lambda e: (len(e["label"]), e["label"]))
    summary = {"figures": len(doc["figures"]),
               "named_figures": sum(1 for f in doc["figures"] if f.get("name")),
               "figure_roles": dict(_count((f.get("role") or {}).get("role", "-") for f in doc["figures"])),
               "figures_skipped": [f["figure_id"] for f in doc["figures"] if f.get("status") == "skipped"],
               "figures_with_reactions": sum(1 for f in doc["figures"] if f.get("n_reactions")),
               "figure_errors": [f["figure_id"] for f in doc["figures"] if f.get("status") not in ("ok", "skipped")],
               "reactions": len(reactions),
               "reactions_from_figures": sum(1 for r in reactions if r.get("source") != "text"),
               "reactions_from_text": sum(1 for r in reactions if r.get("source") == "text"),
               "proposed_steps": sum(1 for r in reactions if r.get("proposed_step")),
               "general_procedures": sum(1 for r in reactions if r.get("general_procedure")),
               "paragraphs": len(doc["paragraphs"]),
               "labels": len(registry), "ambiguous_labels": sum(1 for e in registry.values() if e.get("ambiguous")),
               "named_compounds": len(named), "named_with_structure": sum(1 for e in named.values() if e.get("smiles")),
               "labels_from_names": sum(1 for e in registry.values() if e["origin"] == "named"),
               "labels_from_skipped_figures": sum(1 for e in registry.values() if e["origin"] == "vision"),
               "label_resolution": label_stats,
               "text_items_added": sum(r.get("distinct_added", 0) for r in text_reports.values()),
               "text_items_added_to_reactions": sum(r.get("added", 0) for r in text_reports.values()),
               "text_reactions": {k: (len(v) if k == "dropped" else v) for k, v in (text_report or {}).items()},
               "notes": notes,
               "seconds": round(time.time() - t0, 1)}
    result = {"document": {"files": doc["files"], "model": model_name, "text_model": tm, "out_dir": out_dir,
                           "created": datetime.datetime.now().isoformat(timespec="seconds"),
                           "options": {"use_label_registry": use_label_registry, "use_text_names": use_text_names,
                                       "use_text_context": use_text_context, "use_figure_roles": use_figure_roles,
                                       "skip_spectra": skip_spectra, "use_text_reactions": use_text_reactions,
                                       "dpi": dpi, "large_detector": large_detector}},
              "figures": doc["figures"], "paragraphs": doc["paragraphs"], "compounds": compounds,
              "reactions": reactions, "summary": summary,
              "text_reactions_dropped": (text_report or {}).get("dropped") or []}
    json.dump(result, open(os.path.join(out_dir, "document.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("[pdf] done: %s" % json.dumps(summary, ensure_ascii=False), flush=True)
    return result


def _count(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def main_cli(argv=None):
    ap = argparse.ArgumentParser(description="ChemEagle over a whole PDF (article plus optional SI).")
    ap.add_argument("pdf")
    ap.add_argument("--si", action="append", default=[], help="SI PDF (repeatable)")
    ap.add_argument("--out", default=None, help="output directory (default <pdf>_chemeagle)")
    ap.add_argument("--model", default=None, help="extraction model (default MODEL env, else gemini-3.7-flash)")
    ap.add_argument("--text-model", default=None, help="model of the text-only calls (default: the extraction model)")
    ap.add_argument("--no-text-context", action="store_true")
    ap.add_argument("--no-figure-roles", action="store_true")
    ap.add_argument("--no-skip-spectra", action="store_true", help="run extraction on spectra and data figures too")
    ap.add_argument("--no-text-reactions", action="store_true")
    ap.add_argument("--no-label-registry", action="store_true")
    ap.add_argument("--no-text-names", action="store_true", help="no label structures from compound names (OPSIN)")
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--confidence-llm", action="store_true")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--base-detector", action="store_true", help="VisualHeist base instead of large")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args(argv)
    ChemEagle_PDF(a.pdf, si_paths=a.si, out_dir=a.out, model_name=a.model,
                  use_network=False if a.no_network else None,
                  confidence_llm=True if a.confidence_llm else None,
                  use_label_registry=not a.no_label_registry, use_text_names=not a.no_text_names,
                  use_text_context=not a.no_text_context, use_figure_roles=not a.no_figure_roles,
                  skip_spectra=not a.no_skip_spectra, use_text_reactions=not a.no_text_reactions,
                  text_model=a.text_model, dpi=a.dpi, large_detector=not a.base_detector,
                  resume=not a.no_resume)


if __name__ == "__main__":
    main_cli()
