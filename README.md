# Myanmar Car License Scanner

This repository contains a computer vision pipeline for vehicle and license plate detection, OCR, and result visualization on Myanmar road videos.

It is developed as a **Computer Vision coursework project for CS-8117**.

## Project Overview

The project performs end-to-end automatic number plate recognition (ANPR):

- Detect vehicles in frames.
- Detect license plates for tracked vehicles.
- Read plate text with OCR.
- Visualize and export results for analysis.

## Main Components

- `main.py`: batch/offline processing pipeline.
- `app.py`: Streamlit interface for interactive testing.
- `util.py`: helper functions used across the pipeline.
- `visualize.py`: output visualization utilities.
- `benchmark_latency.py`: latency benchmarking utilities.
- `add_missing_data.py`: post-processing/interpolation helper.

## Models

Pretrained and exported detector models are provided in the repository:

- YOLO model files at project root (`yolov8n.pt`, `yolov8n.onnx`).
- License plate detector models in `models/`.

## Tracking

The repository includes the `sort/` directory as part of this coursework project, since tracking is an important part of the full ANPR pipeline.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For live/app-specific dependencies:

```bash
pip install -r requirements-live.txt
```

## Run

Run the Streamlit app:

```bash
streamlit run app.py
```

Run the main processing script:

```bash
python main.py
```

## Notes

- Large media files and generated outputs are ignored through `.gitignore`.
- If needed, update model paths in scripts to match your local setup.
