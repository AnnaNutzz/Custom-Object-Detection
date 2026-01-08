"""
Evaluation Script

Calculate mAP, FPS, and model size.
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
import cv2
from collections import defaultdict
from tqdm import tqdm

from model import SSDDetector
from dataset import load_voc_dataset, CLASSES


def compute_iou_single(box1, box2):
    """Compute IoU between two boxes"""
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    
    inter = max(0, xmax - xmin) * max(0, ymax - ymin)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter + 1e-8)


def calculate_ap(precision, recall):
    """Calculate Average Precision using 11-point interpolation"""
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    return ap


def evaluate_model(model, test_data, input_size=300, 
                   conf_thresh=0.5, iou_thresh=0.5):
    """
    Evaluate mAP on test set.
    
    Returns:
        mAP, per-class AP dict
    """
    # Collect all predictions and ground truths
    all_predictions = defaultdict(list)  # class -> [(conf, image_id, box)]
    all_ground_truths = defaultdict(list)  # class -> [(image_id, box)]
    
    for img_idx, (img_path, gt_boxes, gt_labels) in enumerate(tqdm(test_data, desc='Evaluating')):
        # Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        image_resized = cv2.resize(image, (input_size, input_size))
        image_input = image_resized.astype(np.float32) / 255.0
        image_input = np.expand_dims(image_input, 0)
        
        # Run inference
        cls_preds, box_preds = model(image_input, training=False)
        
        # Decode predictions
        detections = model.decode_predictions(
            cls_preds, box_preds,
            confidence_threshold=conf_thresh,
            nms_threshold=0.45
        )[0]
        
        # Store ground truths
        for box, label in zip(gt_boxes, gt_labels):
            all_ground_truths[label].append((img_idx, box))
        
        # Store predictions
        if len(detections['boxes']) > 0:
            for box, score, label in zip(
                detections['boxes'].numpy(),
                detections['scores'].numpy(),
                detections['labels'].numpy()
            ):
                all_predictions[label].append((score, img_idx, box))
    
    # Calculate AP for each class
    aps = {}
    
    for cls_id in range(1, len(CLASSES)):  # Skip background
        if cls_id not in all_ground_truths:
            continue
        
        preds = sorted(all_predictions.get(cls_id, []), key=lambda x: -x[0])
        gts = all_ground_truths[cls_id]
        
        # Track which GTs are matched
        gt_matched = np.zeros(len(gts), dtype=bool)
        
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        for pred_idx, (conf, img_id, pred_box) in enumerate(preds):
            # Find matching GT
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_img_id, gt_box) in enumerate(gts):
                if gt_img_id != img_id:
                    continue
                if gt_matched[gt_idx]:
                    continue
                
                iou = compute_iou_single(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_thresh:
                tp[pred_idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_idx] = 1
        
        # Calculate precision/recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / len(gts) if len(gts) > 0 else tp_cumsum
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        ap = calculate_ap(precision, recall)
        aps[CLASSES[cls_id]] = ap
    
    mAP = np.mean(list(aps.values())) if aps else 0
    
    return mAP, aps


def measure_fps(model, input_size=300, num_runs=100, warmup=10):
    """Measure inference FPS"""
    
    # Warmup
    dummy = tf.random.normal((1, input_size, input_size, 3))
    for _ in range(warmup):
        model(dummy, training=False)
    
    # Measure
    start = time.time()
    for _ in range(num_runs):
        model(dummy, training=False)
    elapsed = time.time() - start
    
    fps = num_runs / elapsed
    return fps


def get_model_size(model):
    """Get model size in MB"""
    # Save to temp file
    temp_path = 'temp_model.weights.h5'
    model.save_weights(temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb


def main(args):
    """Main evaluation"""
    
    print("=" * 50)
    print("SSD Object Detector - Evaluation")
    print("=" * 50)
    
    # Load model
    model = SSDDetector(num_classes=6, input_size=300)
    model(tf.random.normal((1, 300, 300, 3)))  # Build
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_weights(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint loaded, using random weights")
    
    # Load test data
    test_data = load_voc_dataset(args.data_path, 'val')
    
    if len(test_data) == 0:
        print("No test data found!")
        return
    
    # Evaluate mAP
    print("\n--- mAP Evaluation ---")
    mAP, aps = evaluate_model(model, test_data)
    
    print(f"\nmAP@0.5: {mAP * 100:.2f}%")
    print("\nPer-class AP:")
    for cls, ap in aps.items():
        print(f"  {cls}: {ap * 100:.2f}%")
    
    # Measure FPS
    print("\n--- FPS Measurement ---")
    fps_gpu = measure_fps(model)
    print(f"FPS (GPU): {fps_gpu:.1f}")
    
    # CPU FPS
    with tf.device('/CPU:0'):
        cpu_model = SSDDetector(num_classes=6, input_size=300)
        cpu_model(tf.random.normal((1, 300, 300, 3)))
        if args.checkpoint and os.path.exists(args.checkpoint):
            cpu_model.load_weights(args.checkpoint)
        fps_cpu = measure_fps(cpu_model)
        print(f"FPS (CPU): {fps_cpu:.1f}")
    
    # Model size
    print("\n--- Model Size ---")
    size_mb = get_model_size(model)
    print(f"Model size: {size_mb:.2f} MB")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"mAP@0.5:      {mAP * 100:.2f}%")
    print(f"FPS (GPU):    {fps_gpu:.1f}")
    print(f"FPS (CPU):    {fps_cpu:.1f}")
    print(f"Model Size:   {size_mb:.2f} MB")
    
    # Save results
    results_path = 'outputs/evaluation_results.txt'
    os.makedirs('outputs', exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(f"mAP@0.5: {mAP * 100:.2f}%\n")
        f.write(f"FPS (GPU): {fps_gpu:.1f}\n")
        f.write(f"FPS (CPU): {fps_cpu:.1f}\n")
        f.write(f"Model Size: {size_mb:.2f} MB\n\n")
        f.write("Per-class AP:\n")
        for cls, ap in aps.items():
            f.write(f"  {cls}: {ap * 100:.2f}%\n")
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SSD Detector')
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/checkpoints/best_model.weights.h5',
                       help='Model checkpoint path')
    parser.add_argument('--data-path', type=str,
                       default='data/VOCdevkit/VOC2007',
                       help='Path to VOC dataset')
    args = parser.parse_args()
    
    main(args)
