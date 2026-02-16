import os
import subprocess
import time
import importlib
from datetime import datetime
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from sort.sort import Sort
from util import get_car, read_license_plate_multi, normalize_plate, accuracy_score

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


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


@st.cache_resource
def load_models(backend: str = "onnx"):
    if backend == "onnx":
        coco_path = "./yolov8n.onnx"
        plate_path = "./models/license_plate_detector.onnx"
    else:
        coco_path = "./yolov8n.pt"
        plate_path = "./models/license_plate_detector.pt"
    coco_model = YOLO(coco_path)
    plate_model = YOLO(plate_path)
    return coco_model, plate_model


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=2, line_length_x=35, line_length_y=35):
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


def _decode_image(uploaded_file) -> np.ndarray:
    data = uploaded_file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def detect_plates(
    bgr: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 640,
    robust_ocr: bool = False,
    debug_ocr: bool = False,
    use_country: bool = True,
) -> List[Dict]:
    _, model = load_models(backend="onnx")
    out = model.predict(bgr, conf=conf, verbose=False, imgsz=imgsz)[0]
    if out.boxes is None or len(out.boxes) == 0:
        return []

    h, w = bgr.shape[:2]
    results = []

    for x1, y1, x2, y2, score, _ in out.boxes.data.tolist():
        bb = clamp_int_bbox(x1, y1, x2, y2, w, h)
        if bb is None:
            continue
        ex = expand_bbox(*bb, w, h)
        if ex is None:
            continue
        x1i, y1i, x2i, y2i = ex
        crop = bgr[y1i:y2i, x1i:x2i]
        if crop is None or crop.size == 0:
            continue
        if debug_ocr:
            text_raw, text_score, dbg = read_license_plate_multi(
                crop, fast=not robust_ocr, debug=True
            )
        else:
            text_raw, text_score = read_license_plate_multi(crop, fast=not robust_ocr)
            dbg = None
        if use_country:
            text, country = normalize_plate(text_raw)
            acc = accuracy_score(text, country)
        else:
            text, country = (text_raw or ""), ""
            acc = 0.0
        combined = float(text_score) * float(score)
        results.append(
            {
                "bbox": [x1i, y1i, x2i, y2i],
                "bbox_score": float(score),
                "raw_text": text_raw,
                "text": text,
                "text_score": float(text_score),
                "combined_score": combined,
                "accuracy_score": acc,
                "country": country,
                "crop": crop,
                "debug": dbg,
            }
        )

    return results


def process_video_file(
    input_video_path: str,
    output_video_path: str,
    backend: str,
    imgsz: int,
    detect_every_n: int,
    plate_detect_every_n: int,
    ocr_every_n: int,
    downscale_width: int,
    coco_conf: float,
    plate_conf: float,
):
    coco_model, plate_model = load_models(backend)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    writer = None
    used_codec = None
    for codec in ("avc1", "H264", "mp4v"):
        test_writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if test_writer.isOpened():
            writer = test_writer
            used_codec = codec
            break

    if writer is None:
        raise RuntimeError("Could not initialize video writer for output video.")

    vehicle_classes = {2, 3, 5, 7}
    tracker = Sort()
    last_coco = None
    last_plate = None
    last_ocr_frame_by_car = {}
    best_text_by_car = {}

    frame_nmr = -1
    frames_processed = 0
    ocr_calls = 0
    start = time.perf_counter()

    while True:
        ret, frame0 = cap.read()
        if not ret:
            break
        frame_nmr += 1
        frames_processed += 1

        orig_h, orig_w = frame0.shape[:2]
        frame = frame0
        if downscale_width and orig_w > downscale_width:
            new_w = downscale_width
            new_h = int(orig_h * (new_w / orig_w))
            frame = cv2.resize(frame0, (new_w, new_h))
            sx = orig_w / new_w
            sy = orig_h / new_h
        else:
            sx = 1.0
            sy = 1.0

        frame_vis = frame0.copy()

        if frame_nmr % detect_every_n == 0 or last_coco is None:
            det_out = coco_model.predict(frame, conf=coco_conf, verbose=False, imgsz=imgsz)[0]
            last_coco = det_out
        else:
            det_out = last_coco

        detections_ = []
        if det_out.boxes is not None and len(det_out.boxes) > 0:
            for x1, y1, x2, y2, score, cls in det_out.boxes.data.tolist():
                if int(cls) in vehicle_classes:
                    detections_.append([x1, y1, x2, y2, score])

        if detections_:
            track_ids = tracker.update(np.asarray(detections_, dtype=np.float32))
        else:
            track_ids = tracker.update(np.empty((0, 5), dtype=np.float32))
        track_list = track_ids.tolist() if len(track_ids) > 0 else []

        for tx1, ty1, tx2, ty2, tcar_id in track_list:
            b = clamp_int_bbox(tx1 * sx, ty1 * sy, tx2 * sx, ty2 * sy, orig_w, orig_h)
            if b is None:
                continue
            x1i, y1i, x2i, y2i = b
            draw_border(frame_vis, (x1i, y1i), (x2i, y2i), color=(0, 255, 0), thickness=2)
            cv2.putText(frame_vis, f"car {int(tcar_id)}", (x1i, max(20, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        if frame_nmr % plate_detect_every_n == 0 or last_plate is None:
            lp_out = plate_model.predict(frame, conf=plate_conf, verbose=False, imgsz=imgsz)[0]
            last_plate = lp_out
        else:
            lp_out = last_plate

        if lp_out.boxes is not None and len(lp_out.boxes) > 0:
            for lp in lp_out.boxes.data.tolist():
                x1, y1, x2, y2, lp_score, cls = lp
                xcar1, ycar1, xcar2, ycar2, car_id = get_car([x1, y1, x2, y2, lp_score, cls], track_list)
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

                crop = frame0[oy1i:oy2i, ox1i:ox2i]
                if crop is None or crop.size == 0:
                    continue

                lastf = last_ocr_frame_by_car.get(car_id, -10**9)
                if (frame_nmr - lastf) >= ocr_every_n:
                    txt_raw, scr = read_license_plate_multi(crop, fast=True)
                    txt, country = normalize_plate(txt_raw)
                    acc = accuracy_score(txt, country)
                    ocr_calls += 1
                    last_ocr_frame_by_car[car_id] = frame_nmr
                    prev = best_text_by_car.get(car_id, ("", "", 0.0, 0.0, "Other"))
                    if scr > prev[2]:
                        best_text_by_car[car_id] = (txt_raw, txt, scr, acc, country)

                best_raw, best_txt, best_scr, best_acc, best_country = best_text_by_car.get(
                    car_id, ("", "", 0.0, 0.0, "Other")
                )
                pbox = clamp_int_bbox(ox1i, oy1i, ox2i, oy2i, orig_w, orig_h)
                if pbox is None:
                    continue
                px1, py1, px2, py2 = pbox
                cv2.rectangle(frame_vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
                label = best_txt if best_txt else "UNKNOWN"
                cv2.putText(frame_vis, label, (px1, max(24, py1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame_vis)

    cap.release()
    writer.release()
    wall = time.perf_counter() - start
    fps_eff = frames_processed / wall if wall > 0 else 0.0

    return {
        "frames": frames_processed,
        "wall_time_s": round(wall, 4),
        "effective_fps": round(fps_eff, 3),
        "ocr_calls": ocr_calls,
        "output_video": output_video_path,
        "writer_codec": used_codec,
    }


def make_streamlit_playable_mp4(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    playable_path = f"{base}_playable.mp4"

    ffmpeg_cmd = None
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            playable_path,
        ]
    except Exception:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            playable_path,
        ]

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(playable_path):
            return playable_path
    except Exception:
        pass

    return input_path


def build_live_processor(
    backend: str,
    imgsz: int,
    detect_every_n: int,
    plate_detect_every_n: int,
    ocr_every_n: int,
    coco_conf: float,
    plate_conf: float,
):
    coco_model, plate_model = load_models(backend)

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.tracker = Sort()
            self.last_coco = None
            self.last_plate = None
            self.last_ocr_frame_by_car = {}
            self.best_text_by_car = {}
            self.frame_nmr = -1
            self.vehicle_classes = {2, 3, 5, 7}

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_nmr += 1
            h, w = img.shape[:2]
            vis = img.copy()

            if self.frame_nmr % detect_every_n == 0 or self.last_coco is None:
                det_out = coco_model.predict(img, conf=coco_conf, verbose=False, imgsz=imgsz)[0]
                self.last_coco = det_out
            else:
                det_out = self.last_coco

            detections_ = []
            if det_out.boxes is not None and len(det_out.boxes) > 0:
                for x1, y1, x2, y2, score, cls in det_out.boxes.data.tolist():
                    if int(cls) in self.vehicle_classes:
                        detections_.append([x1, y1, x2, y2, score])

            if detections_:
                track_ids = self.tracker.update(np.asarray(detections_, dtype=np.float32))
            else:
                track_ids = self.tracker.update(np.empty((0, 5), dtype=np.float32))
            track_list = track_ids.tolist() if len(track_ids) > 0 else []

            for tx1, ty1, tx2, ty2, tcar_id in track_list:
                bb = clamp_int_bbox(tx1, ty1, tx2, ty2, w, h)
                if bb is None:
                    continue
                x1i, y1i, x2i, y2i = bb
                draw_border(vis, (x1i, y1i), (x2i, y2i), color=(0, 255, 0), thickness=2)
                cv2.putText(vis, f"car {int(tcar_id)}", (x1i, max(20, y1i - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            if self.frame_nmr % plate_detect_every_n == 0 or self.last_plate is None:
                lp_out = plate_model.predict(img, conf=plate_conf, verbose=False, imgsz=imgsz)[0]
                self.last_plate = lp_out
            else:
                lp_out = self.last_plate

            if lp_out.boxes is not None and len(lp_out.boxes) > 0:
                for lp in lp_out.boxes.data.tolist():
                    x1, y1, x2, y2, lp_score, cls = lp
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car([x1, y1, x2, y2, lp_score, cls], track_list)
                    if car_id == -1:
                        continue
                    car_id = int(car_id)

                    bb = clamp_int_bbox(x1, y1, x2, y2, w, h)
                    if bb is None:
                        continue
                    ex = expand_bbox(*bb, w, h)
                    if ex is None:
                        continue
                    ox1, oy1, ox2, oy2 = ex
                    crop = img[oy1:oy2, ox1:ox2]
                    if crop is None or crop.size == 0:
                        continue

                    lastf = self.last_ocr_frame_by_car.get(car_id, -10**9)
                    if (self.frame_nmr - lastf) >= ocr_every_n:
                        txt_raw, scr = read_license_plate_multi(crop, fast=True)
                        txt, country = normalize_plate(txt_raw)
                        acc = accuracy_score(txt, country)
                        self.last_ocr_frame_by_car[car_id] = self.frame_nmr
                        prev = self.best_text_by_car.get(car_id, ("", "", 0.0, 0.0, "Other"))
                        if scr > prev[2]:
                            self.best_text_by_car[car_id] = (txt_raw, txt, scr, acc, country)

                    best_raw, best_txt, best_scr, best_acc, best_country = self.best_text_by_car.get(
                        car_id, ("", "", 0.0, 0.0, "Other")
                    )
                    cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                    label = best_txt if best_txt else "UNKNOWN"
                    cv2.putText(vis, label, (ox1, max(24, oy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(vis, format="bgr24")

    return LiveProcessor


def draw_results(bgr: np.ndarray, results: List[Dict]) -> np.ndarray:
    out = bgr.copy()
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        text = r["text"] or "UNKNOWN"
        score = r["text_score"]
        label = f"{text} ({score:.2f})"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            out,
            label,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def _save_crops(results: List[Dict], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, r in enumerate(results, 1):
        crop = r.get("crop")
        if crop is None or crop.size == 0:
            continue
        text = r.get("text", "") or "unknown"
        fname = f"plate_{i:02d}_{text}.png"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, crop)
        saved.append(path)
    return saved


def main():
    st.set_page_config(page_title="Myanmar LPR Studio", layout="wide")
    st.markdown(
        """
        <style>
            .hero {padding: 1rem 1.2rem; border-radius: 14px; background: linear-gradient(120deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; margin-bottom: 1rem;}
            .hero h2 {margin: 0 0 0.2rem 0; color: #f8fafc;}
            .hero p {margin: 0; color: #cbd5e1;}
            .stTabs [data-baseweb="tab-list"] button {font-size: 0.98rem;}
        </style>
        <div class="hero">
            <h2>Myanmar License Plate Recognition Studio</h2>
            <p>Run uploaded video inference, save annotated output, or start real-time webcam inference.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Global Settings")
        backend = st.selectbox("Inference backend", ["onnx", "pytorch"], index=0)
        imgsz = st.selectbox("Inference image size", [320, 512, 640, 768], index=1)
        downscale_width = st.selectbox("Downscale width", [0, 960, 1280, 1600], index=2)
        detect_every_n = st.slider("Vehicle detect every N frames", 1, 6, 3, 1)
        plate_detect_every_n = st.slider("Plate detect every N frames", 1, 6, 2, 1)
        ocr_every_n = st.slider("OCR every N frames / car", 1, 24, 12, 1)
        coco_conf = st.slider("Car detector confidence", 0.10, 0.80, 0.30, 0.01)
        plate_conf = st.slider("Plate detector confidence", 0.05, 0.80, 0.10, 0.01)

    tab_video, tab_live, tab_image = st.tabs(["Video Output", "Live Camera", "Image Test"])

    with tab_video:
        st.subheader("Upload Video → Get Annotated Output")
        up_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], key="video_upload")

        if up_video is not None:
            os.makedirs("uploads", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_path = os.path.join("uploads", f"input_{ts}_{up_video.name}")
            with open(input_path, "wb") as f:
                f.write(up_video.getbuffer())

            st.video(input_path)
            run_video = st.button("Run Full Video Inference", type="primary")
            if run_video:
                out_path = os.path.join("outputs", f"annotated_{ts}.mp4")
                with st.spinner("Running detection + OCR on video..."):
                    stats = process_video_file(
                        input_video_path=input_path,
                        output_video_path=out_path,
                        backend=backend,
                        imgsz=imgsz,
                        detect_every_n=detect_every_n,
                        plate_detect_every_n=plate_detect_every_n,
                        ocr_every_n=ocr_every_n,
                        downscale_width=downscale_width,
                        coco_conf=coco_conf,
                        plate_conf=plate_conf,
                    )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Frames", stats["frames"])
                c2.metric("Total Time (s)", stats["wall_time_s"])
                c3.metric("Effective FPS", stats["effective_fps"])
                c4.metric("OCR Calls", stats["ocr_calls"])
                st.caption(f"Output writer codec: {stats.get('writer_codec', 'unknown')}")

                st.success(f"Annotated video saved: {stats['output_video']}")
                preview_path = make_streamlit_playable_mp4(stats["output_video"])
                if preview_path != stats["output_video"]:
                    st.caption("Showing browser-compatible preview copy.")
                with open(preview_path, "rb") as vf_preview:
                    st.video(vf_preview.read())
                with open(stats["output_video"], "rb") as vf:
                    st.download_button(
                        "Download Annotated Video",
                        data=vf.read(),
                        file_name=os.path.basename(stats["output_video"]),
                        mime="video/mp4",
                    )
        else:
            st.info("Upload one video to run end-to-end inference and export annotated output.")

    with tab_live:
        st.subheader("Real-Time Webcam Inference")
        if not WEBRTC_AVAILABLE:
            st.warning("Live camera requires `streamlit-webrtc` and `av`.")
            st.code("pip install streamlit-webrtc av", language="bash")
        else:
            st.caption("Start webcam stream and see car + plate overlays in real time.")
            live_processor = build_live_processor(
                backend=backend,
                imgsz=imgsz,
                detect_every_n=detect_every_n,
                plate_detect_every_n=plate_detect_every_n,
                ocr_every_n=ocr_every_n,
                coco_conf=coco_conf,
                plate_conf=plate_conf,
            )

            webrtc_streamer(
                key="myanmar-lpr-live",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=live_processor,
                async_processing=True,
            )

    with tab_image:
        st.subheader("Quick Image Plate Test")
        conf = st.slider("Plate detector confidence (image)", 0.05, 0.80, 0.25, 0.01)
        robust = st.checkbox("Robust OCR (slower)", value=True)
        debug_ocr = st.checkbox("Debug OCR (show reason)", value=False)
        use_country = st.checkbox("Use country/format validation", value=True)
        save_crops = st.checkbox("Save detected plate crops", value=False)
        run = st.button("Run Image Detection")

        upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="img_upload")
        if not upload:
            st.info("Upload a photo to start image test.")
        else:
            bgr = _decode_image(upload)
            if bgr is None:
                st.error("Could not decode the image.")
            else:
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.subheader("Input")
                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                if run:
                    results = detect_plates(
                        bgr,
                        conf=conf,
                        imgsz=imgsz,
                        robust_ocr=robust,
                        debug_ocr=debug_ocr,
                        use_country=use_country,
                    )
                    drawn = draw_results(bgr, results)
                    with col1:
                        st.subheader("Annotated")
                        st.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                    with col2:
                        if not results:
                            st.warning("No plates detected.")
                        else:
                            df = pd.DataFrame(
                                [
                                    {
                                        "plate_text_raw": r.get("raw_text", ""),
                                        "plate_text": r["text"],
                                        "text_score": r["text_score"],
                                        "bbox_score": r["bbox_score"],
                                        "bbox": r["bbox"],
                                        "combined_score": r.get("combined_score", 0.0),
                                        "accuracy_score": r.get("accuracy_score", 0.0),
                                        "country": r.get("country", ""),
                                    }
                                    for r in results
                                ]
                            )
                            st.dataframe(df, use_container_width=True)
                            if save_crops:
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                out_dir = os.path.join("outputs", f"plates_{ts}")
                                saved = _save_crops(results, out_dir)
                                if saved:
                                    st.success(f"Saved {len(saved)} crops to `{out_dir}`")
                else:
                    with col2:
                        st.info("Click Run Image Detection.")


if __name__ == "__main__":
    main()
