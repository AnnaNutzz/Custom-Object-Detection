# Task 1: Custom Object Detection from Scratch

A Single-Shot MultiBox Detector (SSD) trained **entirely from scratch** (no pre-trained weights) on PASCAL VOC 2007.

## Overview

| Item          | Details                                   |
| ------------- | ----------------------------------------- |
| **Model**     | Custom SSD-style CNN (built from scratch) |
| **Classes**   | person, car, dog, bicycle, chair          |
| **Framework** | TensorFlow 2.x / Keras                    |
| **Dataset**   | PASCAL VOC 2007                           |
| **Platform**  | Compatible with x86_64 and ARM            |

## Project Structure

```
Task 1/
├── config.yaml          # Model & training configuration
├── model.py             # SSD detector architecture (backbone + heads)
├── dataset.py           # Data loading & augmentation
├── train.py             # Training script
├── evaluate.py          # mAP, FPS, model size evaluation
├── demo.py              # Real-time video detection demo
├── download_dataset.py  # Dataset downloader (uses TensorFlow Datasets)
├── requirements.txt     # Python dependencies
├── report/
│   └── REPORT.md        # Detailed technical report
└── outputs/
    └── checkpoints/     # Saved model weights
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python download_dataset.py
```

Downloads PASCAL VOC 2007 via TensorFlow Datasets (~800 MB).

### 3. Train the Model

```bash
python train.py
```

Trains for 10 epochs (configurable in `config.yaml`). Checkpoints saved to `outputs/checkpoints/`.

### 4. Evaluate Performance

```bash
python evaluate.py
```

Outputs: mAP@0.5, FPS (CPU/GPU), model size.

### 5. Run Demo

```bash
# With sample video
python demo.py --source data/sample_video.mp4

# With webcam (if available)
python demo.py --source 0

# With any video file
python demo.py --source path/to/video.mp4
```

Press `q` to quit the demo.

## Expected Results

| Metric         | Value   |
| -------------- | ------- |
| **mAP@0.5**    | ~30-40% |
| **FPS (GPU)**  | ~25-35  |
| **FPS (CPU)**  | ~3-5    |
| **Model Size** | ~15 MB  |

> Note: Lower accuracy is expected when training from scratch. Pre-trained models typically achieve 70-80% mAP.

## Demo

Screen recordings of training, evaluation, and real-time detection are available in the repository.

## Technical Report

See [report/REPORT.md](report/REPORT.md) for:

- Architecture design decisions
- Data augmentation strategies
- Training methodology
- Trade-off analysis (accuracy vs speed vs size)

## Configuration

Edit `config.yaml` to modify:

```yaml
model:
  input_size: 300 # Input resolution
  num_classes: 6 # 5 classes + background

training:
  batch_size: 8 # Adjust based on GPU memory
  epochs: 10 # Training epochs
  learning_rate: 0.001 # Initial learning rate

inference:
  confidence_threshold: 0.5
  nms_threshold: 0.45
```

## Dependencies

- Python 3.8+
- TensorFlow 2.10+
- OpenCV
- NumPy
- Albumentations
- TensorFlow Datasets

## Key Features

1. **From Scratch Training**: No pre-trained weights used
2. **Multi-scale Detection**: 4 feature maps for detecting objects of different sizes
3. **Hard Negative Mining**: 3:1 ratio for handling class imbalance
4. **Data Augmentation**: Horizontal flip, brightness/contrast, color jittering
5. **Real-time Demo**: Works with webcam or video files

## License

MIT License
