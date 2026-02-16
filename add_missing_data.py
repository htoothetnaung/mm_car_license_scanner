# interpolate.py (FULL UPDATED)
import csv
import numpy as np
from scipy.interpolate import interp1d


def parse_bbox(s):
    s = str(s).strip()
    if not s:
        return None
    s = s.replace("[", "").replace("]", "").strip()
    parts = [p for p in s.split() if p]
    if len(parts) < 4:
        return None
    arr = np.array(list(map(float, parts[:4])), dtype=np.float32)
    if not np.isfinite(arr).all():
        return None
    return arr


def fmt_bbox(a):
    return "[{} {} {} {}]".format(a[0], a[1], a[2], a[3])


def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row["frame_nmr"]) for row in data])
    car_ids = np.array([int(float(row["car_id"])) for row in data])

    car_bboxes = []
    lp_bboxes = []
    for row in data:
        cb = parse_bbox(row["car_bbox"])
        lb = parse_bbox(row["license_plate_bbox"])
        if cb is None or lb is None:
            # skip bad rows
            car_bboxes.append(np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32))
            lp_bboxes.append(np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32))
        else:
            car_bboxes.append(cb)
            lp_bboxes.append(lb)

    car_bboxes = np.array(car_bboxes)
    lp_bboxes = np.array(lp_bboxes)

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        mask = car_ids == car_id
        frames = frame_numbers[mask]
        order = np.argsort(frames)

        frames = frames[order]
        car_seq = car_bboxes[mask][order]
        lp_seq = lp_bboxes[mask][order]

        # remove NaN rows
        ok = np.isfinite(car_seq).all(axis=1) & np.isfinite(lp_seq).all(axis=1)
        frames = frames[ok]
        car_seq = car_seq[ok]
        lp_seq = lp_seq[ok]

        if len(frames) < 2:
            continue

        first_frame = int(frames[0])
        last_frame = int(frames[-1])

        originals = {}
        for row in data:
            if int(float(row["car_id"])) == int(car_id):
                originals[int(row["frame_nmr"])] = row

        full_frames = np.arange(first_frame, last_frame + 1)

        f_car = interp1d(frames, car_seq, axis=0, fill_value="extrapolate")
        f_lp = interp1d(frames, lp_seq, axis=0, fill_value="extrapolate")

        full_car = f_car(full_frames)
        full_lp = f_lp(full_frames)

        last_text = ""
        last_text_score = 0.0

        for i, fr in enumerate(full_frames):
            fr = int(fr)
            row = {
                "frame_nmr": str(fr),
                "car_id": str(car_id),
                "car_bbox": fmt_bbox(full_car[i]),
                "license_plate_bbox": fmt_bbox(full_lp[i]),
            }

            if fr in originals:
                o = originals[fr]
                txt = (o.get("license_number") or "").strip()
                sc = float(o.get("license_number_score") or 0.0)

                if txt != "":
                    last_text = txt
                    last_text_score = sc

                row["license_plate_bbox_score"] = o.get("license_plate_bbox_score", "0")
                row["license_number"] = txt
                row["license_number_score"] = str(sc)
            else:
                row["license_plate_bbox_score"] = "0"
                row["license_number"] = last_text
                row["license_number_score"] = str(last_text_score if last_text else 0.0)

            interpolated_data.append(row)

    return interpolated_data


if __name__ == "__main__":
    IN_CSV = "test4.csv"
    OUT_CSV = "test4_interpolated.csv"

    with open(IN_CSV, "r") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    out = interpolate_bounding_boxes(data)

    header = [
        "frame_nmr",
        "car_id",
        "car_bbox",
        "license_plate_bbox",
        "license_plate_bbox_score",
        "license_number",
        "license_number_score",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out)

    print("Saved:", OUT_CSV)
