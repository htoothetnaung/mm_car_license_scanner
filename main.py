from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime

from sort.sort import Sort
from util import get_car, normalize_plate, read_license_plate_multi, write_csv, accuracy_score


def _safe_output_csv(path: str) -> str:
    base, ext = os.path.splitext(path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}{ext or '.csv'}"


def clamp_int_bbox(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w, round(x2))))
    y2 = int(max(0, min(h, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, w, h, mx=0.18, my=0.30):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    dx = int(round(mx * bw))
    dy = int(round(my * bh))
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _print_ocr_debug(frame_nmr: int, car_id: int, dbg: dict) -> None:
    reason = dbg.get("reason", "unknown")
    w = dbg.get("w")
    h = dbg.get("h")
    blur_var = dbg.get("blur_var")
    blur_recovery = dbg.get("blur_recovery")
    print(
        f"[frame {frame_nmr}] car_id={car_id} OCR UNKNOWN reason={reason} "
        f"crop={w}x{h} min_size=40x12 blur_var={blur_var} blur_thresh=80.0 "
        f"blur_recovery={blur_recovery}"
    )

    rows = dbg.get("rows") or []
    if not rows:
        return

    rows = sorted(rows, key=lambda r: r.get("adjusted", 0.0), reverse=True)[:10]
    header = f"{'scale':>5} {'var':>3} {'score':>6} {'adj':>6} {'len':>3} {'A':>2} {'D':>2} text"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['scale']:>5.2f} {r['variant']:>3} {r['score']:>6.3f} {r['adjusted']:>6.3f} "
            f"{r['len']:>3} {r['alpha']:>2} {r['digit']:>2} {r['text']}"
        )


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=3, line_length_x=40, line_length_y=40):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def main():
    VIDEO_PATH = "./myanmar_license_video/IMG_9704.MP4"
    OUT_CSV = "./test4.csv"
    OUT_VIDEO = "./outputs/live_overlay.mp4"
    DEBUG_OCR = True
    SAVE_VIDEO = True
    SHOW_PREVIEW = False

    MAX_FRAMES = None
    DOWNSCALE_WIDTH = 1280

    # ✅ smaller imgsz is faster
    YOLO_IMGSZ = 512

    # ✅ run detectors less often (reuse cached results)
    DETECT_EVERY_N_FRAMES = 3
    PLATE_DETECT_EVERY_N_FRAMES = 2

    VEHICLE_CLASSES = {2, 3, 5, 7}
    COCO_CONF = 0.30
    PLATE_CONF = 0.10

    # ✅ OCR less often per car
    OCR_EVERY_N_FRAMES = 12

    results = {}
    mot_tracker = Sort()

    coco_model = YOLO("yolov8n.pt")
    plate_model = YOLO("./models/license_plate_detector.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    writer = None
    if SAVE_VIDEO:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(os.path.dirname(OUT_VIDEO) or ".", exist_ok=True)
        writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    best_text_by_car = {}          # car_id -> (raw_text, text, score)
    last_ocr_frame_by_car = {}     # car_id -> last OCR frame

    frame_nmr = -1
    last_coco = None
    last_plate = None

    while True:
        ret, frame0 = cap.read()
        if not ret:
            break

        frame_nmr += 1
        if MAX_FRAMES is not None and frame_nmr >= MAX_FRAMES:
            break

        orig_h, orig_w = frame0.shape[:2]

        # downscale for detection
        frame = frame0
        if DOWNSCALE_WIDTH and orig_w > DOWNSCALE_WIDTH:
            new_w = DOWNSCALE_WIDTH
            new_h = int(orig_h * (new_w / orig_w))
            frame = cv2.resize(frame0, (new_w, new_h))
            sx = orig_w / new_w
            sy = orig_h / new_h
        else:
            sx = 1.0
            sy = 1.0

        frame_vis = frame0.copy() if (SAVE_VIDEO or SHOW_PREVIEW) else None

        results[frame_nmr] = {}

        # --- COCO detect (cached) ---
        if frame_nmr % DETECT_EVERY_N_FRAMES == 0 or last_coco is None:
            det_out = coco_model.predict(frame, conf=COCO_CONF, verbose=False, imgsz=YOLO_IMGSZ)[0]
            last_coco = det_out
        else:
            det_out = last_coco

        detections_ = []
        if det_out.boxes is not None and len(det_out.boxes) > 0:
            for x1, y1, x2, y2, score, cls in det_out.boxes.data.tolist():
                if int(cls) in VEHICLE_CLASSES:
                    detections_.append([x1, y1, x2, y2, score])

        # --- SORT tracking ---
        if detections_:
            track_ids = mot_tracker.update(np.asarray(detections_, dtype=np.float32))
        else:
            track_ids = mot_tracker.update(np.empty((0, 5), dtype=np.float32))
        track_list = track_ids.tolist() if len(track_ids) > 0 else []

        if frame_vis is not None:
            for tx1, ty1, tx2, ty2, tcar_id in track_list:
                bx = clamp_int_bbox(tx1 * sx, ty1 * sy, tx2 * sx, ty2 * sy, orig_w, orig_h)
                if bx is None:
                    continue
                x1i, y1i, x2i, y2i = bx
                draw_border(frame_vis, (x1i, y1i), (x2i, y2i), color=(0, 255, 0), thickness=2)
                cv2.putText(
                    frame_vis,
                    f"car {int(tcar_id)}",
                    (x1i, max(20, y1i - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # --- Plate detect (cached) ---
        if frame_nmr % PLATE_DETECT_EVERY_N_FRAMES == 0 or last_plate is None:
            lp_out = plate_model.predict(frame, conf=PLATE_CONF, verbose=False, imgsz=YOLO_IMGSZ)[0]
            last_plate = lp_out
        else:
            lp_out = last_plate

        if lp_out.boxes is None or len(lp_out.boxes) == 0:
            if frame_nmr % 60 == 0:
                print(f"[frame {frame_nmr}] tracks={len(track_list)} plates=0")
        else:
            for lp in lp_out.boxes.data.tolist():
                x1, y1, x2, y2, lp_score, cls = lp

                xcar1, ycar1, xcar2, ycar2, car_id = get_car([x1, y1, x2, y2, lp_score, cls], track_list)
                if car_id == -1:
                    continue
                car_id = int(car_id)

                # plate crop from original frame
                ox1, oy1, ox2, oy2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                ob = clamp_int_bbox(ox1, oy1, ox2, oy2, orig_w, orig_h)
                if ob is None:
                    continue
                ox1i, oy1i, ox2i, oy2i = ob

                ob2 = expand_bbox(ox1i, oy1i, ox2i, oy2i, orig_w, orig_h)
                if ob2 is None:
                    continue
                ox1i, oy1i, ox2i, oy2i = ob2

                crop0 = frame0[oy1i:oy2i, ox1i:ox2i]
                if crop0 is None or crop0.size == 0:
                    continue
                # OCR throttling
                lastf = last_ocr_frame_by_car.get(car_id, -10**9)
                if (frame_nmr - lastf) >= OCR_EVERY_N_FRAMES:
                    if DEBUG_OCR:
                        txt_raw, scr, dbg = read_license_plate_multi(crop0, fast=True, debug=True)
                    else:
                        txt_raw, scr = read_license_plate_multi(crop0, fast=True)
                        dbg = None
                    txt, country = normalize_plate(txt_raw)
                    acc = accuracy_score(txt, country)
                    last_ocr_frame_by_car[car_id] = frame_nmr
                    prev_raw, prev_txt, prev_scr, prev_acc, prev_country = best_text_by_car.get(
                        car_id, ("", "", 0.0, 0.0, "Other")
                    )
                    if scr > prev_scr:
                        best_text_by_car[car_id] = (txt_raw, txt, scr, acc, country)
                    if DEBUG_OCR and dbg is not None and not txt_raw:
                        _print_ocr_debug(frame_nmr, car_id, dbg)

                best_raw, best_txt, best_scr, best_acc, best_country = best_text_by_car.get(
                    car_id, ("", "", 0.0, 0.0, "Other")
                )

                car_bbox = [xcar1 * sx, ycar1 * sy, xcar2 * sx, ycar2 * sy]
                plate_bbox = [ox1i, oy1i, ox2i, oy2i]

                results[frame_nmr][car_id] = {
                    "car": {"bbox": car_bbox},
                    "license_plate": {
                        "bbox": plate_bbox,
                        "raw_text": best_raw or "",
                        "text": best_txt or "",
                        "bbox_score": float(lp_score),
                        "text_score": float(best_scr),
                        "combined_score": float(best_scr) * float(lp_score),
                        "accuracy_score": float(best_acc),
                        "country": best_country,
                    },
                }

                if frame_vis is not None:
                    pbox = clamp_int_bbox(ox1i, oy1i, ox2i, oy2i, orig_w, orig_h)
                    if pbox is not None:
                        px1, py1, px2, py2 = pbox
                        cv2.rectangle(frame_vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        label = best_txt if best_txt else "UNKNOWN"
                        cv2.putText(
                            frame_vis,
                            label,
                            (px1, max(24, py1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

        if frame_nmr % 60 == 0:
            print(f"[frame {frame_nmr}] tracks={len(track_list)} plates={len(lp_out.boxes)}")

        if frame_vis is not None and writer is not None:
            writer.write(frame_vis)
        if frame_vis is not None and SHOW_PREVIEW:
            cv2.imshow("LPR Live", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print("✅ Video saved:", OUT_VIDEO)
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    try:
        write_csv(results, OUT_CSV)
        print("✅ CSV saved:", OUT_CSV)
    except PermissionError:
        alt = _safe_output_csv(OUT_CSV)
        write_csv(results, alt)
        print("⚠️ CSV saved:", alt, "(because Excel locked the old file)")


if __name__ == "__main__":
    main()
