"""
Training Script for SSD Object Detector

Train custom SSD from scratch on PASCAL VOC.
"""

import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from model import SSDDetector, SSDLoss
from dataset import load_voc_dataset, load_voc_tfds, VOCDataGenerator, CLASSES


def setup_gpu():
    """Configure GPU memory growth if available"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found - training on CPU")


def train(config_path='config.yaml'):
    """Main training function"""
    
    # Setup
    setup_gpu()
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("SSD Object Detector - Training from Scratch")
    print("=" * 50)
    
    # Parameters
    input_size = config['model']['input_size']
    num_classes = config['model']['num_classes']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    data_path = config['data']['dataset_path']
    
    print(f"\nConfig:")
    print(f"  Input size: {input_size}x{input_size}")
    print(f"  Classes: {config['data']['classes']}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    
    # Create model
    print("\nBuilding model...")
    model = SSDDetector(num_classes=num_classes, input_size=input_size)
    
    # Build model
    sample = tf.random.normal((1, input_size, input_size, 3))
    model(sample)
    model.summary()
    
    # Load dataset - try TFDS first, then traditional format
    print(f"\nLoading dataset...")
    use_tfds = False
    
    # Try TFDS format first (data/voc folder)
    train_data = load_voc_tfds('data', 'train')
    val_data_tfds = load_voc_tfds('data', 'validation')
    
    if train_data:
        use_tfds = True
        # Combine train and validation for more data
        train_data = train_data + val_data_tfds
    else:
        # Fallback to traditional VOC format
        print(f"Trying traditional format at {data_path}...")
        train_data = load_voc_dataset(data_path, 'trainval')
    
    if len(train_data) == 0:
        print("ERROR: No data found! Run: python download_dataset.py")
        return
    
    # Split into train/val
    np.random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)
    train_set = train_data[:split_idx]
    val_set = train_data[split_idx:]
    
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")
    
    # Create data generators
    train_gen = VOCDataGenerator(
        train_set, model.anchors, 
        batch_size=batch_size, 
        input_size=input_size,
        augment=True,
        use_tfds=use_tfds
    )
    
    val_gen = VOCDataGenerator(
        val_set, model.anchors,
        batch_size=batch_size,
        input_size=input_size,
        augment=False,
        shuffle=False,
        use_tfds=use_tfds
    )
    
    # Loss and optimizer
    loss_fn = SSDLoss(num_classes=num_classes)
    
    optimizer = keras.optimizers.SGD(
        learning_rate=lr,
        momentum=0.9,
        weight_decay=config['training']['weight_decay']
    )
    
    # Training metrics
    train_loss_metric = keras.metrics.Mean(name='train_loss')
    val_loss_metric = keras.metrics.Mean(name='val_loss')
    
    # Checkpoint directory
    checkpoint_dir = 'outputs/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # TensorBoard
    log_dir = f'outputs/logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Training step
    def train_step(images, targets):
        with tf.GradientTape() as tape:
            cls_preds, box_preds = model(images, training=True)
            loss = loss_fn((targets[0], targets[1]), (cls_preds, box_preds))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # Validation step
    def val_step(images, targets):
        cls_preds, box_preds = model(images, training=False)
        loss = loss_fn((targets[0], targets[1]), (cls_preds, box_preds))
        return loss
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    start_epoch = config.get('resume', {}).get('start_epoch', 0)
    
    # Resume from checkpoint if specified
    resume_path = config.get('resume', {}).get('checkpoint_path', None)
    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming from checkpoint: {resume_path}")
        print(f"Starting from epoch {start_epoch + 1}")
        model.load_weights(resume_path)
        best_val_loss = config.get('resume', {}).get('best_val_loss', float('inf'))
        print(f"Previous best val_loss: {best_val_loss:.4f}")
    
    import time
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss_metric.reset_state()
        val_loss_metric.reset_state()
        
        # Train
        progress = tf.keras.utils.Progbar(len(train_gen))
        for batch_idx, (images, targets) in enumerate(train_gen):
            loss = train_step(images, targets)
            train_loss_metric.update_state(loss)
            progress.update(batch_idx + 1, [('loss', loss.numpy())])
        
        train_loss = train_loss_metric.result().numpy()
        
        # Validate (now compiled and faster)
        for images, targets in val_gen:
            val_loss = val_step(images, targets)
            val_loss_metric.update_state(val_loss)
        
        val_loss = val_loss_metric.result().numpy()
        epoch_time = time.time() - epoch_start
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Log to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('loss/train', train_loss, step=epoch)
            tf.summary.scalar('loss/val', val_loss, step=epoch)
            tf.summary.scalar('time/epoch', epoch_time, step=epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(f'{checkpoint_dir}/best_model.weights.h5')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.save_weights(f'{checkpoint_dir}/epoch_{epoch+1}.weights.h5')
        
        # Reset generators
        train_gen.on_epoch_end()
    
    # Save final model
    model.save_weights(f'{checkpoint_dir}/final_model.weights.h5')
    print(f"\nTraining complete! Model saved to {checkpoint_dir}")
    print(f"TensorBoard logs: {log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SSD Detector')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)
