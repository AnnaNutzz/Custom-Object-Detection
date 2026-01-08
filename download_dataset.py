"""
Download PASCAL VOC Dataset

Downloads VOC2007 using TensorFlow Datasets (downloads from Google servers).
Falls back to alternative mirrors if needed.
"""

import os
import sys

# Target classes we care about
TARGET_CLASSES = ['person', 'car', 'dog', 'bicycle', 'chair']


def download_voc_tfds(data_dir='data'):
    """Download PASCAL VOC 2007 using TensorFlow Datasets"""
    
    try:
        import tensorflow_datasets as tfds
        print("Using TensorFlow Datasets to download VOC2007...")
        print("(Downloads from Google servers)\n")
        
        # Download VOC2007
        ds, info = tfds.load(
            'voc/2007',
            split='train+validation',
            data_dir=data_dir,
            with_info=True,
            download=True
        )
        
        print(f"\nDataset downloaded successfully!")
        print(f"Location: {data_dir}")
        print(f"Total examples: {info.splits['train'].num_examples + info.splits['validation'].num_examples}")
        print(f"\nClasses: {info.features['objects']['label'].names}")
        
        return True
        
    except ImportError:
        print("tensorflow_datasets not installed. Installing...")
        os.system(f"{sys.executable} -m pip install tensorflow-datasets")
        print("Please run this script again.")
        return False
    except Exception as e:
        print(f"TFDS download failed: {e}")
        return False


def create_sample_dataset(data_dir='data'):
    """Create a minimal sample dataset for testing if download fails"""
    
    import numpy as np
    
    print("\nCreating sample dataset for testing...")
    
    sample_dir = os.path.join(data_dir, 'sample_dataset')
    images_dir = os.path.join(sample_dir, 'images')
    labels_dir = os.path.join(sample_dir, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Create 50 sample images with random boxes
    try:
        import cv2
        
        for i in range(50):
            # Create a random colored image
            img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Add some shapes to make it more realistic
            for _ in range(3):
                x1 = np.random.randint(50, 500)
                y1 = np.random.randint(50, 400)
                x2 = x1 + np.random.randint(50, 150)
                y2 = y1 + np.random.randint(50, 100)
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            # Save image
            img_path = os.path.join(images_dir, f'sample_{i:04d}.jpg')
            cv2.imwrite(img_path, img)
            
            # Create corresponding label (YOLO format: class x_center y_center width height)
            label_path = os.path.join(labels_dir, f'sample_{i:04d}.txt')
            with open(label_path, 'w') as f:
                # Add 1-3 random boxes
                for _ in range(np.random.randint(1, 4)):
                    cls = np.random.randint(0, 5)  # 5 classes
                    x_center = np.random.uniform(0.2, 0.8)
                    y_center = np.random.uniform(0.2, 0.8)
                    width = np.random.uniform(0.1, 0.3)
                    height = np.random.uniform(0.1, 0.3)
                    f.write(f"{cls} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")
        
        print(f"Created {50} sample images at: {sample_dir}")
        print("NOTE: This is synthetic data for testing only!")
        return True
        
    except ImportError:
        print("OpenCV not available for sample creation")
        return False


def download_voc(data_dir='data'):
    """Main download function"""
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 50)
    print("PASCAL VOC Dataset Downloader")
    print("=" * 50)
    
    # Check if already exists (TFDS format)
    tfds_path = os.path.join(data_dir, 'voc')
    if os.path.exists(tfds_path):
        print(f"Dataset may already exist at: {tfds_path}")
    
    # Try TFDS first (uses Google servers)
    if download_voc_tfds(data_dir):
        print("\n✓ Download complete!")
        return
    
    # If TFDS fails, create sample data
    print("\n" + "=" * 50)
    print("Could not download real dataset.")
    print("Creating sample dataset for testing...")
    print("=" * 50)
    
    if create_sample_dataset(data_dir):
        print("\n✓ Sample dataset created!")
        print("You can use this for testing the training pipeline.")
    else:
        print("\n✗ Failed to create sample dataset.")
        print("\nManual alternatives:")
        print("1. Download VOC2007 from Kaggle:")
        print("   https://www.kaggle.com/datasets/zaraks/pascal-voc-2007")
        print("2. Use COCO dataset subset from TensorFlow")
        sys.exit(1)


if __name__ == '__main__':
    download_voc()
