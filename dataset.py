"""
PASCAL VOC Dataset Loader

Supports both TensorFlow Datasets format and traditional VOC folder structure.
"""

import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import albumentations as A
from tqdm import tqdm


# Default classes
CLASSES = ["background", "person", "car", "dog", "bicycle", "chair"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# VOC class names mapping (TFDS uses these indices)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
VOC_TO_OUR_CLASS = {
    'person': 1, 'car': 2, 'dog': 3, 'bicycle': 4, 'chair': 5
}


def parse_voc_annotation(xml_path, classes=CLASSES):
    """Parse PASCAL VOC XML annotation"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    boxes = []
    labels = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_TO_IDX:
            continue
        
        # Skip difficult objects
        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text) / width
        ymin = float(bbox.find('ymin').text) / height
        xmax = float(bbox.find('xmax').text) / width
        ymax = float(bbox.find('ymax').text) / height
        
        # Validate
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])
    
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int32)


def load_voc_tfds(data_path='data', split='train'):
    """
    Load PASCAL VOC from TensorFlow Datasets format.
    
    Args:
        data_path: Path to data directory containing 'voc' folder
        split: 'train', 'validation', or 'test'
    
    Returns:
        List of (image, boxes, labels) tuples
    """
    try:
        import tensorflow_datasets as tfds
        
        print(f"Loading VOC from TensorFlow Datasets ({split})...")
        
        ds = tfds.load(
            'voc/2007',
            split=split,
            data_dir=data_path
        )
        
        dataset = []
        
        for example in tqdm(ds, desc=f'Loading {split}'):
            image = example['image'].numpy()
            
            # Get bounding boxes and labels
            objects = example['objects']
            bbox = objects['bbox'].numpy()  # [ymin, xmin, ymax, xmax] normalized
            voc_labels = objects['label'].numpy()  # VOC class indices
            
            # Convert to our format
            boxes = []
            labels = []
            
            for i, voc_label in enumerate(voc_labels):
                voc_class_name = VOC_CLASSES[voc_label]
                
                if voc_class_name in VOC_TO_OUR_CLASS:
                    our_label = VOC_TO_OUR_CLASS[voc_class_name]
                    
                    # Convert bbox from [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
                    ymin, xmin, ymax, xmax = bbox[i]
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(our_label)
            
            if len(boxes) > 0:
                dataset.append((
                    image,  # RGB numpy array
                    np.array(boxes, dtype=np.float32),
                    np.array(labels, dtype=np.int32)
                ))
        
        print(f"Loaded {len(dataset)} images for {split}")
        return dataset
        
    except Exception as e:
        print(f"TFDS loading failed: {e}")
        return []


def load_voc_dataset(data_path, split='train'):
    """
    Load PASCAL VOC dataset (traditional folder structure).
    
    Args:
        data_path: Path to VOC2007 directory
        split: 'train', 'val', or 'trainval'
    
    Returns:
        List of (image_path, boxes, labels) tuples
    """
    split_file = os.path.join(data_path, 'ImageSets', 'Main', f'{split}.txt')
    
    if not os.path.exists(split_file):
        print(f"Split file not found: {split_file}")
        return []
    
    with open(split_file) as f:
        image_ids = [line.strip() for line in f if line.strip()]
    
    dataset = []
    
    for img_id in tqdm(image_ids, desc=f'Loading {split}'):
        img_path = os.path.join(data_path, 'JPEGImages', f'{img_id}.jpg')
        xml_path = os.path.join(data_path, 'Annotations', f'{img_id}.xml')
        
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue
        
        boxes, labels = parse_voc_annotation(xml_path)
        
        # Only keep images with our target classes
        if len(boxes) > 0:
            dataset.append((img_path, boxes, labels))
    
    print(f"Loaded {len(dataset)} images for {split}")
    return dataset


def get_augmentations(training=True):
    """Get data augmentation pipeline"""
    if training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.1),
        ], bbox_params=A.BboxParams(
            format='albumentations',  # [xmin, ymin, xmax, ymax] normalized
            label_fields=['labels'],
            min_visibility=0.3
        ))
    else:
        return None


class VOCDataGenerator:
    """
    Data generator for training.
    
    Loads images, applies augmentation, and matches anchors to ground truth.
    """
    
    def __init__(self, dataset, anchors, batch_size=8, input_size=300, 
                 augment=True, shuffle=True, use_tfds=False):
        self.dataset = dataset
        self.anchors = anchors
        self.batch_size = batch_size
        self.input_size = input_size
        self.augment = augment
        self.shuffle = shuffle
        self.use_tfds = use_tfds
        self.transforms = get_augmentations(augment)
        
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __iter__(self):
        self.current = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.current >= len(self):
            raise StopIteration
        
        batch_indices = self.indices[self.current * self.batch_size:
                                     (self.current + 1) * self.batch_size]
        self.current += 1
        
        return self._load_batch(batch_indices)
    
    def _load_batch(self, indices):
        """Load and preprocess a batch"""
        from model import match_anchors_to_gt
        
        images = []
        cls_targets = []
        box_targets = []
        
        for idx in indices:
            data = self.dataset[idx]
            
            if self.use_tfds:
                # TFDS format: (image_array, boxes, labels)
                image, boxes, labels = data
                image = image.copy()  # Already RGB numpy array
            else:
                # Traditional format: (image_path, boxes, labels)
                img_path, boxes, labels = data
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            if self.transforms is not None:
                try:
                    transformed = self.transforms(
                        image=image, 
                        bboxes=boxes.tolist(), 
                        labels=labels.tolist()
                    )
                    image = transformed['image']
                    boxes = np.array(transformed['bboxes'], dtype=np.float32)
                    labels = np.array(transformed['labels'], dtype=np.int32)
                except:
                    pass  # Keep original if augmentation fails
            
            # Resize
            image = cv2.resize(image, (self.input_size, self.input_size))
            image = image.astype(np.float32) / 255.0
            
            # Match anchors
            cls_target, box_target = match_anchors_to_gt(
                self.anchors, boxes, labels
            )
            
            images.append(image)
            cls_targets.append(cls_target)
            box_targets.append(box_target)
        
        return (
            np.array(images),
            (np.array(cls_targets), np.array(box_targets))
        )
    
    def on_epoch_end(self):
        """Shuffle at epoch end"""
        if self.shuffle:
            np.random.shuffle(self.indices)


# Quick test
if __name__ == '__main__':
    # Try TFDS first
    dataset = load_voc_tfds('data', 'train')
    
    if not dataset:
        # Fallback to traditional format
        data_path = 'data/VOCdevkit/VOC2007'
        if os.path.exists(data_path):
            dataset = load_voc_dataset(data_path, 'trainval')
    
    if dataset:
        print(f"\nLoaded {len(dataset)} images")
        
        # Show class distribution
        from collections import Counter
        all_labels = []
        for data in dataset:
            labels = data[2]  # labels is the 3rd element
            all_labels.extend(labels.tolist())
        
        print("\nClass distribution:")
        for label, count in sorted(Counter(all_labels).items()):
            print(f"  {CLASSES[label]}: {count}")
    else:
        print("No dataset found!")
        print("Run: python download_dataset.py")
