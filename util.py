import os
# ---- Silence Paddle logs (MUST be before importing paddleocr) ----
os.environ["FLAGS_log_level"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["KMP_WARNINGS"] = "0"

import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)

import csv
import re
import numbers
import numpy as np
import cv2
from paddleocr import PaddleOCR

# ✅ recognition-only (NO detector) + no angle cls for speed
ocr = PaddleOCR(
    lang="en",
    det=False,
    rec=True,
    use_angle_cls=False,
)

UK_REGEX = r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$"
MM_REGEX = r"^[0-9][A-Z][0-9]{4}$"
SG_REGEX = r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]$"

_UK_RE = re.compile(UK_REGEX)
_MM_RE = re.compile(MM_REGEX)
_SG_RE = re.compile(SG_REGEX)

_DIGIT_MAP = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
    "T": "7",
}

_ALPHA_MAP = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
    "7": "T",
}

def write_csv(results: dict, output_path: str) -> None:
    header = [
        "frame_nmr",
        "car_id",
        "car_bbox",
        "license_plate_bbox",
        "license_plate_bbox_score",
        "license_number_raw",
        "license_number",
        "license_number_score",
        "combined_score",
        "accuracy_score",
        "country",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for frame_nmr in sorted(results.keys()):
            for car_id in sorted(results[frame_nmr].keys()):
                row = results[frame_nmr][car_id]
                if "car" not in row or "license_plate" not in row:
                    continue
                car_bbox = row["car"].get("bbox", None)
                lp = row["license_plate"]
                lp_bbox = lp.get("bbox", None)
                if car_bbox is None or lp_bbox is None:
                    continue

                w.writerow(
                    [
                        frame_nmr,
                        car_id,
                        f"[{car_bbox[0]} {car_bbox[1]} {car_bbox[2]} {car_bbox[3]}]",
                        f"[{lp_bbox[0]} {lp_bbox[1]} {lp_bbox[2]} {lp_bbox[3]}]",
                        float(lp.get("bbox_score", 0.0)),
                        lp.get("raw_text", "") or "",
                        lp.get("text", "") or "",
                        float(lp.get("text_score", 0.0)),
                        float(lp.get("combined_score", 0.0)),
                        float(lp.get("accuracy_score", 0.0)),
                        lp.get("country", "") or "",
                    ]
                )

def _clean_text(text: str) -> str:
    text = (text or "").upper().strip()
    return re.sub(r"[^A-Z0-9]", "", text)

def _clean_raw_text(text: str) -> str:
    raw = (text or "").upper().strip()
    raw = re.sub(r"[^A-Z0-9\- ]", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()

def _has_uk_space(text: str) -> bool:
    raw = (text or "").upper()
    raw = re.sub(r"[^A-Z0-9 ]", " ", raw)
    parts = [p for p in raw.split() if p]
    return len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 3

def _has_mm_hyphen(text: str) -> bool:
    raw = (text or "").upper()
    raw = re.sub(r"[^A-Z0-9-]", "", raw)
    return bool(re.fullmatch(r"[0-9][A-Z]-[0-9]{4}", raw))

def _apply_pattern(text: str, pattern: str) -> str:
    if len(text) != len(pattern):
        return text
    out = []
    for ch, p in zip(text, pattern):
        if p == "D":
            if ch.isdigit():
                out.append(ch)
            else:
                out.append(_DIGIT_MAP.get(ch, ch))
        else:  # "L"
            if ch.isalpha():
                out.append(ch)
            else:
                out.append(_ALPHA_MAP.get(ch, ch))
    return "".join(out)

def _normalize_sg(text: str) -> str:
    n = len(text)
    if n < 3 or n > 8:
        return text
    # L{1,3} D{1,4} L (total length 3..8)
    for l1 in range(1, 4):
        for d1 in range(1, 5):
            if l1 + d1 + 1 != n:
                continue
            pattern = "L" * l1 + "D" * d1 + "L"
            cand = _apply_pattern(text, pattern)
            if _SG_RE.match(cand):
                # Insert space before digits for SG format later in normalize_plate.
                return cand
    return text

def _looks_like_sg_input(text: str) -> bool:
    """
    Guard: SG plates must end with a letter (check letter).
    Avoid "correcting" non-SG plates like 'BIG918'.
    """
    raw = (text or "").upper()
    raw = re.sub(r"[^A-Z0-9]", "", raw)
    if len(raw) < 3 or len(raw) > 8:
        return False
    if not raw[-1].isalpha():
        return False
    # Must have at least one digit before the last letter
    return any(c.isdigit() for c in raw[:-1])

def _sg_checksum_letter(prefix_letters: str, digits: str) -> str:
    """
    Singapore vehicle plate checksum.
    Uses last two prefix letters (or 0 + letter for single-letter prefix),
    4-digit number (left-padded with zeros), weights [9,4,5,4,3,2],
    remainder mapped to letters "A Z Y X U T S R P M L K J H G E D C B".
    """
    prefix_letters = (prefix_letters or "").upper()
    digits = re.sub(r"\D", "", digits or "")
    digits = digits.zfill(4)[-4:]

    if len(prefix_letters) >= 2:
        p1 = prefix_letters[-2]
        p2 = prefix_letters[-1]
    elif len(prefix_letters) == 1:
        p1 = "0"
        p2 = prefix_letters[0]
    else:
        p1 = "0"
        p2 = "0"

    def _val(ch: str) -> int:
        if ch == "0":
            return 0
        if "A" <= ch <= "Z":
            return ord(ch) - ord("A") + 1
        return 0

    seq = [_val(p1), _val(p2)] + [int(d) for d in digits]
    weights = [9, 4, 5, 4, 3, 2]
    total = sum(a * b for a, b in zip(seq, weights))
    remainder = total % 19
    mapping = "AZYXUTSRPMLKJHGEDCB"
    return mapping[remainder]

def normalize_plate(text: str) -> tuple[str, str]:
    raw = _clean_raw_text(text)
    t = _clean_text(text)
    has_space = bool(raw and " " in raw)
    if not t and not raw:
        return "", "Other"

    # Length-gated matching + correction
    if len(t) == 7:
        cand = _apply_pattern(t, "LLDDLLL")
        if _UK_RE.match(cand) or _has_uk_space(text):
            # Only keep space if OCR/raw had a space.
            if has_space or _has_uk_space(text):
                return f"{cand[:4]} {cand[4:]}", "UK"
            return cand, "UK"
        if has_space:
            return raw, "Other"
        return t, "Other"

    if len(t) == 6:
        if _has_mm_hyphen(text):
            cand = _apply_pattern(t, "DLDDDD")
            if _MM_RE.match(cand):
                # Keep hyphen if OCR/raw had it.
                if "-" in _clean_raw_text(text):
                    return f"{cand[:2]}-{cand[2:]}", "Myanmar"
                return cand, "Myanmar"
        # If it otherwise matches Myanmar pattern, enforce hyphen output.
        cand = _apply_pattern(t, "DLDDDD")
        if _MM_RE.match(cand):
            return f"{cand[:2]}-{cand[2:]}", "Myanmar"

    cand = _normalize_sg(t)
    if _looks_like_sg_input(text) and _SG_RE.match(cand):
        prefix = re.sub(r"[^A-Z]", "", cand[:-1])
        digits = re.sub(r"[^0-9]", "", cand)
        check = cand[-1]
        expected = _sg_checksum_letter(prefix, digits)
        # If checksum doesn't match, correct it (OCR often misreads the last letter).
        if expected and check != expected:
            cand = cand[:-1] + expected
        # Only insert space if OCR/raw had a space.
        if has_space:
            m = re.match(r"^([A-Z]{1,3})([0-9]{1,4})([A-Z])$", cand)
            if m:
                cand = f"{m.group(1)} {m.group(2)}{m.group(3)}"
        return cand, "Singapore"

    if has_space:
        return raw, "Other"
    return t, "Other"

def plate_country(text: str) -> str:
    _, country = normalize_plate(text)
    return country

def accuracy_score(text: str, country: str) -> float:
    """
    Heuristic accuracy score (0..1) without ground truth.
    """
    raw = _clean_raw_text(text)
    t = _clean_text(text)
    if not t and not raw:
        return 0.0

    score = 0.0

    # length sanity
    if 4 <= len(t) <= 10:
        score += 0.2

    # mixed alpha-numeric
    if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
        score += 0.2

    # pattern match bonus
    t_nospace = t
    if _UK_RE.match(t_nospace):
        score += 0.4
    elif _MM_RE.match(t_nospace):
        score += 0.4
    elif _SG_RE.match(t_nospace):
        score += 0.4

    # SG checksum bonus if applicable
    if country == "Singapore" and _SG_RE.match(t_nospace):
        prefix = re.sub(r"[^A-Z]", "", t_nospace[:-1])
        digits = re.sub(r"[^0-9]", "", t_nospace)
        check = t_nospace[-1]
        expected = _sg_checksum_letter(prefix, digits)
        if expected and check == expected:
            score += 0.2

    return min(1.0, max(0.0, score))

def _looks_like_plate(text: str) -> bool:
    t = _clean_text(text)
    if len(t) < 4 or len(t) > 10:
        return False
    return any(c.isdigit() for c in t) and (sum(c.isalpha() for c in t) >= 2)

def _score_bias(text: str) -> float:
    """
    Prefer mixed alpha-numeric plates over digits-only OCR.
    """
    t = _clean_text(text)
    alpha = sum(c.isalpha() for c in t)
    digit = sum(c.isdigit() for c in t)
    if alpha == 0 and digit > 0:
        return 0.55
    bias = 1.0
    if alpha >= 1 and digit >= 1:
        bias *= 1.10
    if alpha >= 2 and digit >= 2:
        bias *= 1.10
    # Strongly prefer more letters (helps keep full prefixes like "SJG")
    bias *= 1.0 + min(0.40, 0.10 * alpha)
    return bias

def _is_blurry(bgr: np.ndarray, thresh: float = 80.0) -> bool:
    """
    Simple blur detector using variance of Laplacian.
    Lower variance => blurrier image.
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    v = cv2.Laplacian(gray, cv2.CV_64F).var()
    return v < thresh

def _extract_pairs(out):
    """
    PaddleOCR det=False returns different formats depending on version.
    Normalize to list of (text, score).
    """
    pairs = []

    if not out:
        return pairs

    def _is_pair(x):
        return (
            isinstance(x, (list, tuple))
            and len(x) == 2
            and isinstance(x[0], str)
            and isinstance(x[1], numbers.Real)
        )

    def _walk(x):
        if _is_pair(x):
            pairs.append((x[0], float(x[1])))
            return
        if isinstance(x, (list, tuple)):
            for item in x:
                _walk(item)

    _walk(out)
    return pairs

def _prep_fast_variants(bgr: np.ndarray):
    """
    FAST variants only (2 images): CLAHE gray + inverted CLAHE gray (as BGR)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    v1 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    v2 = cv2.cvtColor(255 - g, cv2.COLOR_GRAY2BGR)
    return [v1, v2]


def _prep_robust_variants(bgr: np.ndarray, blur_recovery: bool = False):
    """
    More OCR-friendly variants (still limited count).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(g, -1, kernel)

    # adaptive threshold
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    )

    variants = [
        cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(255 - g, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(th, cv2.COLOR_GRAY2BGR),
    ]
    if blur_recovery:
        dn = cv2.fastNlMeansDenoising(g, h=12, templateWindowSize=7, searchWindowSize=21)
        unsharp = cv2.addWeighted(g, 1.6, cv2.GaussianBlur(g, (0, 0), 1.2), -0.6, 0)
        kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]], dtype=np.float32)
        deblur = cv2.filter2D(g, -1, kernel)
        variants.extend(
            [
                cv2.cvtColor(dn, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(deblur, cv2.COLOR_GRAY2BGR),
            ]
        )
    return variants

def read_license_plate_multi(plate_bgr, fast: bool = True, blur_recovery: bool = False, debug: bool = False):
    """
    PaddleOCR recognition-only over a cropped plate region.
    Returns (text, score). Never None.
    """
    if plate_bgr is None or getattr(plate_bgr, "size", 0) == 0:
        return ("", 0.0, {"reason": "empty_crop"}) if debug else ("", 0.0)

    h, w = plate_bgr.shape[:2]

    # ✅ skip tiny crops (saves huge time)
    min_w, min_h = 40, 12
    blur_thresh = 80.0
    if w < min_w or h < min_h:
        if debug:
            return "", 0.0, {
                "reason": "too_small",
                "w": w,
                "h": h,
                "min_w": min_w,
                "min_h": min_h,
                "blur_thresh": blur_thresh,
            }
        return "", 0.0

    # auto-enable blur recovery if needed
    blur_var = None
    if not blur_recovery:
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_var < blur_thresh:
            blur_recovery = True

    # upscale small plates
    base_scale = 3.0 if max(h, w) < 140 else 2.0
    scales = [base_scale] if fast else [base_scale, base_scale * 1.3, base_scale * 1.6]
    if blur_recovery and not fast:
        scales.append(base_scale * 2.0)

    best_text, best_score = "", 0.0
    debug_rows = []

    for scale in scales:
        plate = cv2.resize(
            plate_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

        variants = _prep_fast_variants(plate) if fast else _prep_robust_variants(plate, blur_recovery=blur_recovery)

        for v_idx, img in enumerate(variants):
            # PaddleOCR expects RGB input
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # IMPORTANT: no cls=True (your version may not support it)
            out = ocr.ocr(img_rgb)
            for text, score in _extract_pairs(out):
                t = _clean_text(text)
                if len(t) < 4:
                    continue
                adjusted = float(score) * (1.15 if _looks_like_plate(t) else 1.0)
                adjusted *= _score_bias(text)
                # prefer longer text slightly to reduce partial reads
                adjusted *= 1.0 + min(0.25, 0.04 * max(0, len(t) - 6))
                if debug:
                    alpha = sum(c.isalpha() for c in t)
                    digit = sum(c.isdigit() for c in t)
                    debug_rows.append(
                        {
                            "scale": scale,
                            "variant": v_idx,
                            "text": _clean_raw_text(text),
                            "score": float(score),
                            "adjusted": float(adjusted),
                            "len": len(t),
                            "alpha": alpha,
                            "digit": digit,
                        }
                    )
                if adjusted > best_score:
                    best_text, best_score = _clean_raw_text(text), adjusted

    if debug:
        return best_text, float(best_score), {
            "reason": "ok" if best_text else "no_candidates",
            "w": w,
            "h": h,
            "min_w": min_w,
            "min_h": min_h,
            "blur_var": blur_var,
            "blur_thresh": blur_thresh,
            "blur_recovery": blur_recovery,
            "rows": debug_rows,
        }
    return best_text, float(best_score)

def read_license_plate_strict(plate_bgr):
    """
    Single-pass OCR for the strict enhancement flow:
    resize 2x -> grayscale -> CLAHE -> OCR -> best text/score.
    """
    if plate_bgr is None or getattr(plate_bgr, "size", 0) == 0:
        return "", 0.0

    h, w = plate_bgr.shape[:2]
    if w < 40 or h < 12:
        return "", 0.0

    plate = cv2.resize(plate_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    img_rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    out = ocr.ocr(img_rgb)
    best_text, best_score = "", 0.0
    for text, score in _extract_pairs(out):
        t = _clean_text(text)
        if len(t) < 4:
            continue
        adjusted = float(score) * (1.15 if _looks_like_plate(t) else 1.0)
        adjusted *= _score_bias(text)
        adjusted *= 1.0 + min(0.25, 0.04 * max(0, len(t) - 6))
        if adjusted > best_score:
            best_text, best_score = _clean_raw_text(text), adjusted

    return best_text, float(best_score)

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def get_car(license_plate_box6, vehicle_track_ids):
    x1, y1, x2, y2, *_ = license_plate_box6
    plate_box = [x1, y1, x2, y2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    best = None
    best_area = None
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if cx >= xcar1 and cx <= xcar2 and cy >= ycar1 and cy <= ycar2:
            area = (xcar2 - xcar1) * (ycar2 - ycar1)
            if best is None or area < best_area:
                best = (xcar1, ycar1, xcar2, ycar2, car_id)
                best_area = area
    if best is not None:
        return best

    best_iou = 0.0
    best = None
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        i = _iou(plate_box, [xcar1, ycar1, xcar2, ycar2])
        if i > best_iou:
            best_iou = i
            best = (xcar1, ycar1, xcar2, ycar2, car_id)

    if best is not None and best_iou > 0.01:
        return best
    return -1, -1, -1, -1, -1
