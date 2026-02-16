import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from sort.sort import Sort
from util import get_car, normalize_plate, read_license_plate_multi


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


def ensure_onnx(model_pt_path: str, output_name: str, imgsz: int) -> str:
    out_path = Path(output_name)
    if out_path.exists():
        return str(out_path)

    model = YOLO(model_pt_path)
    exported = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
    exported_path = Path(exported)
    if exported_path.resolve() != out_path.resolve():
        if out_path.exists():
            out_path.unlink()
        exported_path.replace(out_path)
    return str(out_path)


def benchmark_video(video_path: str, coco_model: YOLO, plate_model: YOLO, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    mot_tracker = Sort()
    last_coco = None
    last_plate = None
    last_ocr_frame_by_car = {}

    frame_nmr = -1
    processed_frames = 0
    ocr_calls = 0
    ocr_success = 0

    coco_ms = 0.0
    plate_ms = 0.0
    track_ms = 0.0
    ocr_ms = 0.0
    assoc_ms = 0.0

    start_wall = time.perf_counter()

    while True:
        ret, frame0 = cap.read()
        if not ret:
            break

        frame_nmr += 1
        if args.max_frames is not None and frame_nmr >= args.max_frames:
            break
        processed_frames += 1

        orig_h, orig_w = frame0.shape[:2]
        frame = frame0
        if args.downscale_width and orig_w > args.downscale_width:
            new_w = args.downscale_width
            new_h = int(orig_h * (new_w / orig_w))
            frame = cv2.resize(frame0, (new_w, new_h))
            sx = orig_w / new_w
            sy = orig_h / new_h
        else:
            sx = 1.0
            sy = 1.0

        if frame_nmr % args.detect_every_n == 0 or last_coco is None:
            t0 = time.perf_counter()
            det_out = coco_model.predict(frame, conf=args.coco_conf, verbose=False, imgsz=args.imgsz)[0]
            coco_ms += (time.perf_counter() - t0) * 1000.0
            last_coco = det_out
        else:
            det_out = last_coco

        detections_ = []
        if det_out.boxes is not None and len(det_out.boxes) > 0:
            for x1, y1, x2, y2, score, cls in det_out.boxes.data.tolist():
                if int(cls) in args.vehicle_classes:
                    detections_.append([x1, y1, x2, y2, score])

        t0 = time.perf_counter()
        if detections_:
            track_ids = mot_tracker.update(np.asarray(detections_, dtype=np.float32))
        else:
            track_ids = mot_tracker.update(np.empty((0, 5), dtype=np.float32))
        track_ms += (time.perf_counter() - t0) * 1000.0

        track_list = track_ids.tolist() if len(track_ids) > 0 else []

        if frame_nmr % args.plate_detect_every_n == 0 or last_plate is None:
            t0 = time.perf_counter()
            lp_out = plate_model.predict(frame, conf=args.plate_conf, verbose=False, imgsz=args.imgsz)[0]
            plate_ms += (time.perf_counter() - t0) * 1000.0
            last_plate = lp_out
        else:
            lp_out = last_plate

        if lp_out.boxes is None or len(lp_out.boxes) == 0:
            continue

        for lp in lp_out.boxes.data.tolist():
            x1, y1, x2, y2, lp_score, cls = lp

            t0 = time.perf_counter()
            xcar1, ycar1, xcar2, ycar2, car_id = get_car([x1, y1, x2, y2, lp_score, cls], track_list)
            assoc_ms += (time.perf_counter() - t0) * 1000.0
            if car_id == -1:
                continue
            car_id = int(car_id)

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

            lastf = last_ocr_frame_by_car.get(car_id, -10**9)
            if (frame_nmr - lastf) < args.ocr_every_n:
                continue

            t0 = time.perf_counter()
            txt_raw, scr = read_license_plate_multi(crop0, fast=True)
            ocr_ms += (time.perf_counter() - t0) * 1000.0
            ocr_calls += 1
            last_ocr_frame_by_car[car_id] = frame_nmr

            txt, country = normalize_plate(txt_raw)
            if txt and country == "Myanmar":
                ocr_success += 1

    wall_s = time.perf_counter() - start_wall
    cap.release()

    fps_eff = (processed_frames / wall_s) if wall_s > 0 else 0.0
    ocr_success_rate = (ocr_success / ocr_calls) if ocr_calls > 0 else 0.0

    return {
        "video": os.path.basename(video_path),
        "frames_processed": processed_frames,
        "video_fps": native_fps,
        "video_frames_total": native_frames,
        "wall_time_s": round(wall_s, 4),
        "effective_fps": round(fps_eff, 3),
        "coco_infer_total_ms": round(coco_ms, 2),
        "plate_infer_total_ms": round(plate_ms, 2),
        "track_total_ms": round(track_ms, 2),
        "ocr_total_ms": round(ocr_ms, 2),
        "assoc_total_ms": round(assoc_ms, 2),
        "ocr_calls": ocr_calls,
        "ocr_success": ocr_success,
        "ocr_success_rate": round(ocr_success_rate, 4),
        "backend": args.backend,
        "imgsz": args.imgsz,
        "detect_every_n": args.detect_every_n,
        "plate_detect_every_n": args.plate_detect_every_n,
        "ocr_every_n": args.ocr_every_n,
        "downscale_width": args.downscale_width,
    }


def find_videos(videos_dir: str, single_video: str | None = None):
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    if single_video:
        p = Path(single_video)
        if not p.exists():
            raise RuntimeError(f"Video file not found: {single_video}")
        if p.suffix.lower() not in video_exts:
            raise RuntimeError(f"Unsupported video extension: {p.suffix}")
        return [str(p)]

    p = Path(videos_dir)
    files = sorted([str(x) for x in p.glob("*") if x.suffix.lower() in video_exts])
    return files


def main():
    parser = argparse.ArgumentParser(description="Benchmark Myanmar LPR pipeline latency")
    parser.add_argument("--videos-dir", default="myanmar_license_video")
    parser.add_argument("--video", default=None, help="Run benchmark on a single video file")
    parser.add_argument("--backend", choices=["pytorch", "onnx"], required=True)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--detect-every-n", type=int, default=3)
    parser.add_argument("--plate-detect-every-n", type=int, default=2)
    parser.add_argument("--ocr-every-n", type=int, default=12)
    parser.add_argument("--downscale-width", type=int, default=1280)
    parser.add_argument("--coco-conf", type=float, default=0.30)
    parser.add_argument("--plate-conf", type=float, default=0.10)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        args.max_frames = None

    args.vehicle_classes = {2, 3, 5, 7}

    if args.backend == "pytorch":
        coco_path = "yolov8n.pt"
        plate_path = "models/license_plate_detector.pt"
    else:
        coco_path = ensure_onnx("yolov8n.pt", "yolov8n.onnx", args.imgsz)
        plate_path = ensure_onnx("models/license_plate_detector.pt", "models/license_plate_detector.onnx", args.imgsz)

    print(f"Loading models backend={args.backend}: {coco_path}, {plate_path}")
    coco_model = YOLO(coco_path)
    plate_model = YOLO(plate_path)

    videos = find_videos(args.videos_dir, args.video)
    if not videos:
        raise RuntimeError(f"No videos found in: {args.videos_dir}")

    rows = []
    for idx, video_path in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] Benchmarking {video_path}")
        row = benchmark_video(video_path, coco_model, plate_model, args)
        rows.append(row)
        print(
            f"  -> FPS={row['effective_fps']}, wall={row['wall_time_s']}s, "
            f"ocr_calls={row['ocr_calls']}, ocr_success_rate={row['ocr_success_rate']}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.out_csv or f"outputs/benchmark_{args.backend}_{ts}.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_fps = sum(r["effective_fps"] for r in rows) / len(rows)
    print(f"Saved: {out_csv}")
    print(f"Average effective FPS ({args.backend}): {avg_fps:.3f}")


if __name__ == "__main__":
    main()
