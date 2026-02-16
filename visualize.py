# visualize.py (FULL UPDATED)
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=120, line_length_y=120):
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


def parse_bbox(val):
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.replace("[", "").replace("]", "").strip()
    parts = [p for p in s.split() if p]
    if len(parts) < 4:
        return None
    arr = np.array(list(map(float, parts[:4])), dtype=float)
    if not np.isfinite(arr).all():
        return None
    return tuple(arr.tolist())


def clamp_bbox(b, w, h):
    if b is None:
        return None
    x1, y1, x2, y2 = b
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main():
    CSV_PATH = "./test4_interpolated.csv"
    VIDEO_PATH = "./sample1.mp4"
    OUT_VIDEO = "./out4.mp4"

    df = pd.read_csv(CSV_PATH)

    # Fix NaN text → empty string
    if "license_number" in df.columns:
        df["license_number"] = df["license_number"].fillna("").astype(str)
        df.loc[df["license_number"].str.lower() == "nan", "license_number"] = ""
    else:
        df["license_number"] = ""

    if "license_number_score" in df.columns:
        df["license_number_score"] = pd.to_numeric(df["license_number_score"], errors="coerce").fillna(0.0)
    else:
        df["license_number_score"] = 0.0

    df["frame_nmr"] = pd.to_numeric(df["frame_nmr"], errors="coerce").fillna(-1).astype(int)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # pick best plate text per car
    best_text_by_car = {}
    best_row_by_car = {}

    for car_id in df["car_id"].dropna().unique():
        d = df[df["car_id"] == car_id]
        if len(d) == 0:
            continue
        idx = d["license_number_score"].idxmax()
        best_row_by_car[car_id] = df.loc[idx]
        best_text_by_car[car_id] = str(df.loc[idx].get("license_number", "") or "")

    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        rows = df[df["frame_nmr"] == frame_nmr]
        if len(rows) == 0:
            out.write(frame)
            continue

        h, w = frame.shape[:2]

        for _, r in rows.iterrows():
            car_bbox = clamp_bbox(parse_bbox(r.get("car_bbox", None)), w, h)
            lp_bbox = clamp_bbox(parse_bbox(r.get("license_plate_bbox", None)), w, h)
            if car_bbox is None or lp_bbox is None:
                continue

            car_x1, car_y1, car_x2, car_y2 = car_bbox
            x1, y1, x2, y2 = lp_bbox

            draw_border(frame, (car_x1, car_y1), (car_x2, car_y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            car_id = r.get("car_id", None)
            text = best_text_by_car.get(car_id, "")
            if text:
                cv2.putText(frame, text, (car_x1, max(30, car_y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    cap.release()
    print("✅ Saved:", OUT_VIDEO)


if __name__ == "__main__":
    main()
