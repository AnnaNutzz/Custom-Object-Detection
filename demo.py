"""
Real-time Detection Demo

Runs detection on video and saves output video file.
"""

import os
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf

from model import SSDDetector
from dataset import CLASSES


# Colors for each class (BGR)
COLORS = [
    (0, 0, 0),       # background
    (0, 255, 0),     # person - green
    (255, 0, 0),     # car - blue
    (0, 255, 255),   # dog - yellow
    (255, 0, 255),   # bicycle - magenta
    (0, 165, 255),   # chair - orange
]


def draw_detections(image, detections, classes=CLASSES, colors=COLORS):
    """Draw bounding boxes on image"""
    h, w = image.shape[:2]
    
    boxes = detections['boxes'].numpy()
    scores = detections['scores'].numpy()
    labels = detections['labels'].numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        if label >= len(classes):
            continue
        
        # Convert normalized coords to pixels
        xmin = int(box[0] * w)
        ymin = int(box[1] * h)
        xmax = int(box[2] * w)
        ymax = int(box[3] * h)
        
        color = colors[label % len(colors)]
        
        # Draw box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw label
        text = f"{classes[label]}: {score:.2f}"
        font_scale = 0.6
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        cv2.rectangle(
            image, 
            (xmin, ymin - text_h - 10), 
            (xmin + text_w, ymin), 
            color, -1
        )
        cv2.putText(
            image, text, (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )
    
    return image


def run_demo(args):
    """Run detection demo and save output video"""
    
    print("=" * 50)
    print("SSD Object Detector - Demo")
    print("=" * 50)
    
    # Load model
    model = SSDDetector(num_classes=6, input_size=300)
    model(tf.random.normal((1, 300, 300, 3)))  # Build
    
    if os.path.exists(args.checkpoint):
        model.load_weights(args.checkpoint)
        print(f"Loaded: {args.checkpoint}")
    else:
        print("WARNING: Using random weights!")
    
    # Open video source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Failed to open: {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Source: {source} ({width}x{height} @ {fps_video:.1f} FPS)")
    print(f"Total frames: {total_frames}")
    
    # Setup output video
    os.makedirs('outputs', exist_ok=True)
    output_path = args.output or 'outputs/demo_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
    
    print(f"Output: {output_path}")
    print("\nProcessing...")
    
    frame_count = 0
    fps_display = 0
    fps_timer = time.time()
    total_inference_time = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (300, 300))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, 0)
        
        # Inference
        start = time.time()
        cls_preds, box_preds = model(input_img, training=False)
        inference_time = time.time() - start
        total_inference_time += inference_time
        
        # Decode
        detections = model.decode_predictions(
            cls_preds, box_preds,
            confidence_threshold=args.conf_thresh,
            nms_threshold=args.nms_thresh
        )[0]
        
        # Draw
        output_frame = draw_detections(frame.copy(), detections)
        
        # FPS counter
        frame_count += 1
        processed_frames += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = frame_count
            frame_count = 0
            fps_timer = time.time()
        
        cv2.putText(
            output_frame, f"FPS: {fps_display}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # Write to output video
        out.write(output_frame)
        
        # Progress
        if processed_frames % 50 == 0:
            progress = processed_frames / total_frames * 100 if total_frames > 0 else 0
            print(f"  Processed: {processed_frames}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    # Stats
    avg_fps = processed_frames / total_inference_time if total_inference_time > 0 else 0
    
    print("\n" + "=" * 50)
    print("Demo Complete!")
    print("=" * 50)
    print(f"Frames processed: {processed_frames}")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Output saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSD Detection Demo')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0=webcam, or video path)')
    parser.add_argument('--output', type=str, default='outputs/demo_output.mp4',
                       help='Output video path')
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/checkpoints/best_model.weights.h5',
                       help='Model checkpoint')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--nms-thresh', type=float, default=0.45,
                       help='NMS threshold')
    
    args = parser.parse_args()
    run_demo(args)
