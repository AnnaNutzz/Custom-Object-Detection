"""
Custom SSD-Style Object Detector - Built from Scratch

A single-shot multi-box detector with:
- Custom CNN backbone (no pre-trained weights)
- Multi-scale feature maps for detection
- Anchor-based predictions

Architecture:
    Input (300x300x3)
        ↓
    Backbone CNN (5 conv blocks)
        ↓
    Multi-scale Feature Maps
        ↓
    Detection Heads (class + box predictions)
        ↓
    NMS → Final Detections
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


# =============================================================================
# BACKBONE CNN (From Scratch)
# =============================================================================

def conv_block(x, filters, kernel_size=3, strides=1, name=None):
    """Standard conv block: Conv -> BatchNorm -> ReLU"""
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, 
        padding='same', use_bias=False, name=f'{name}_conv'
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.ReLU(name=f'{name}_relu')(x)
    return x


def build_backbone(input_shape=(300, 300, 3)):
    """
    Custom VGG-style backbone trained from scratch.
    
    Returns multiple feature maps at different scales for multi-scale detection.
    
    Feature Maps:
        - conv4: 38x38 (small objects)
        - conv5: 19x19 (medium objects)
        - conv6: 10x10 (medium-large objects)
        - conv7: 5x5 (large objects)
    """
    inputs = keras.Input(shape=input_shape, name='image_input')
    
    # Block 1: 300 -> 150
    x = conv_block(inputs, 32, 3, name='conv1_1')
    x = conv_block(x, 32, 3, name='conv1_2')
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    # Block 2: 150 -> 75
    x = conv_block(x, 64, 3, name='conv2_1')
    x = conv_block(x, 64, 3, name='conv2_2')
    x = layers.MaxPooling2D(2, name='pool2')(x)
    
    # Block 3: 75 -> 38 (Feature Map 1)
    x = conv_block(x, 128, 3, name='conv3_1')
    x = conv_block(x, 128, 3, name='conv3_2')
    x = layers.MaxPooling2D(2, strides=2, padding='same', name='pool3')(x)
    
    # Block 4: 38x38 - Feature Map 1 (for small objects)
    x = conv_block(x, 256, 3, name='conv4_1')
    x = conv_block(x, 256, 3, name='conv4_2')
    feature_map_1 = x  # 38x38x256
    x = layers.MaxPooling2D(2, name='pool4')(x)
    
    # Block 5: 19x19 - Feature Map 2 (for medium objects)
    x = conv_block(x, 512, 3, name='conv5_1')
    x = conv_block(x, 512, 3, name='conv5_2')
    feature_map_2 = x  # 19x19x512
    x = layers.MaxPooling2D(2, padding='same', name='pool5')(x)
    
    # Block 6: 10x10 - Feature Map 3
    x = conv_block(x, 512, 3, name='conv6_1')
    x = conv_block(x, 512, 3, name='conv6_2')
    feature_map_3 = x  # 10x10x512
    x = layers.MaxPooling2D(2, name='pool6')(x)
    
    # Block 7: 5x5 - Feature Map 4 (for large objects)
    x = conv_block(x, 256, 3, name='conv7_1')
    x = conv_block(x, 256, 3, name='conv7_2')
    feature_map_4 = x  # 5x5x256
    
    return Model(inputs, [feature_map_1, feature_map_2, feature_map_3, feature_map_4], 
                 name='backbone')


# =============================================================================
# ANCHOR BOXES
# =============================================================================

def generate_anchors(feature_map_sizes, image_size=300, 
                     scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88],
                     aspect_ratios=[1.0, 2.0, 0.5]):
    """
    Generate anchor boxes for each feature map.
    
    Args:
        feature_map_sizes: List of (height, width) for each feature map
        image_size: Input image size
        scales: Anchor scales relative to image size
        aspect_ratios: Anchor aspect ratios (width/height)
    
    Returns:
        anchors: Array of shape (total_anchors, 4) in [cx, cy, w, h] format
    """
    all_anchors = []
    
    for idx, (fh, fw) in enumerate(feature_map_sizes):
        # Scale for this feature map
        scale = scales[idx] if idx < len(scales) else scales[-1]
        scale_next = scales[idx + 1] if idx + 1 < len(scales) else 1.0
        
        for i in range(fh):
            for j in range(fw):
                # Center of anchor (normalized 0-1)
                cx = (j + 0.5) / fw
                cy = (i + 0.5) / fh
                
                # Add anchors with different aspect ratios
                for ratio in aspect_ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    all_anchors.append([cx, cy, w, h])
                
                # Add extra anchor with geometric mean of scales
                w = h = np.sqrt(scale * scale_next)
                all_anchors.append([cx, cy, w, h])
    
    return np.array(all_anchors, dtype=np.float32)


# =============================================================================
# COMPLETE SSD MODEL
# =============================================================================

class SSDDetector(Model):
    """
    Single Shot MultiBox Detector (SSD) - Built from Scratch
    
    Key components:
    1. Custom CNN backbone for feature extraction
    2. Multi-scale feature maps for detecting objects of different sizes
    3. Anchor-based predictions with class scores and box offsets
    4. Non-Maximum Suppression for final detections
    """
    
    def __init__(self, num_classes=6, input_size=300, **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes  # Including background
        self.input_size = input_size
        self.num_anchors_per_cell = 4  # 3 aspect ratios + 1 extra
        
        # Feature map sizes for 300x300 input
        self.feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5)]
        
        # Build backbone
        self.backbone = build_backbone((input_size, input_size, 3))
        
        # Detection heads for each feature map
        self.cls_heads = []
        self.box_heads = []
        
        for idx, (fh, fw) in enumerate(self.feature_map_sizes):
            self.cls_heads.append(
                keras.Sequential([
                    layers.Conv2D(256, 3, padding='same', activation='relu'),
                    layers.Conv2D(self.num_anchors_per_cell * num_classes, 3, padding='same'),
                ], name=f'cls_head_{idx}')
            )
            self.box_heads.append(
                keras.Sequential([
                    layers.Conv2D(256, 3, padding='same', activation='relu'),
                    layers.Conv2D(self.num_anchors_per_cell * 4, 3, padding='same'),
                ], name=f'box_head_{idx}')
            )
        
        # Generate anchors
        self.anchors = generate_anchors(self.feature_map_sizes, input_size)
        self.num_anchors = len(self.anchors)
        print(f"Total anchors: {self.num_anchors}")
    
    def call(self, images, training=False):
        """Forward pass"""
        feature_maps = self.backbone(images, training=training)
        
        all_cls_preds = []
        all_box_preds = []
        
        for idx, fm in enumerate(feature_maps):
            batch_size = tf.shape(fm)[0]
            
            cls_pred = self.cls_heads[idx](fm)
            cls_pred = tf.reshape(cls_pred, [batch_size, -1, self.num_classes])
            all_cls_preds.append(cls_pred)
            
            box_pred = self.box_heads[idx](fm)
            box_pred = tf.reshape(box_pred, [batch_size, -1, 4])
            all_box_preds.append(box_pred)
        
        cls_preds = tf.concat(all_cls_preds, axis=1)
        box_preds = tf.concat(all_box_preds, axis=1)
        
        return cls_preds, box_preds
    
    def decode_predictions(self, cls_preds, box_preds, 
                          confidence_threshold=0.5, nms_threshold=0.45):
        """Decode predictions to bounding boxes"""
        batch_size = tf.shape(cls_preds)[0]
        cls_probs = tf.nn.softmax(cls_preds, axis=-1)
        
        all_detections = []
        
        for b in range(batch_size):
            probs = cls_probs[b]
            offsets = box_preds[b]
            
            boxes = self._decode_boxes(offsets)
            
            class_probs = probs[:, 1:]  # Exclude background
            scores = tf.reduce_max(class_probs, axis=-1)
            labels = tf.argmax(class_probs, axis=-1) + 1
            
            mask = scores > confidence_threshold
            boxes = tf.boolean_mask(boxes, mask)
            scores = tf.boolean_mask(scores, mask)
            labels = tf.boolean_mask(labels, mask)
            
            final_boxes, final_scores, final_labels = self._nms_per_class(
                boxes, scores, labels, nms_threshold
            )
            
            all_detections.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            })
        
        return all_detections
    
    def _decode_boxes(self, offsets):
        """Decode box offsets relative to anchors"""
        anchors = tf.constant(self.anchors)
        
        cx = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
        cy = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
        w = anchors[:, 2] * tf.exp(offsets[:, 2])
        h = anchors[:, 3] * tf.exp(offsets[:, 3])
        
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        
        boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
        boxes = tf.clip_by_value(boxes, 0.0, 1.0)
        
        return boxes
    
    def _nms_per_class(self, boxes, scores, labels, nms_threshold):
        """Apply NMS per class"""
        unique_labels = tf.unique(labels)[0]
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for label in unique_labels:
            mask = labels == label
            class_boxes = tf.boolean_mask(boxes, mask)
            class_scores = tf.boolean_mask(scores, mask)
            
            indices = tf.image.non_max_suppression(
                class_boxes, class_scores, 
                max_output_size=100, 
                iou_threshold=nms_threshold
            )
            
            all_boxes.append(tf.gather(class_boxes, indices))
            all_scores.append(tf.gather(class_scores, indices))
            all_labels.append(tf.fill([tf.shape(indices)[0]], label))
        
        if len(all_boxes) > 0:
            return (tf.concat(all_boxes, axis=0),
                    tf.concat(all_scores, axis=0),
                    tf.concat(all_labels, axis=0))
        else:
            return (tf.zeros((0, 4)), tf.zeros((0,)), tf.zeros((0,), dtype=tf.int64))


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class SSDLoss(keras.losses.Loss):
    """SSD Loss = Classification Loss + Localization Loss"""
    
    def __init__(self, num_classes=6, neg_pos_ratio=3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
    
    def smooth_l1_loss(self, y_true, y_pred):
        """Compute smooth L1 loss element-wise"""
        diff = tf.abs(y_true - y_pred)
        less_than_one = tf.cast(diff < 1.0, tf.float32)
        loss = less_than_one * 0.5 * diff ** 2 + (1.0 - less_than_one) * (diff - 0.5)
        return loss
    
    def call(self, y_true, y_pred):
        """
        Compute SSD loss.
        
        Args:
            y_true: Tuple of (cls_targets, box_targets)
                - cls_targets: [batch, num_anchors] with class indices
                - box_targets: [batch, num_anchors, 4] with box offsets
            y_pred: Tuple of (cls_preds, box_preds)
                - cls_preds: [batch, num_anchors, num_classes] logits
                - box_preds: [batch, num_anchors, 4] predicted offsets
        """
        cls_targets, box_targets = y_true
        cls_preds, box_preds = y_pred
        
        # Ensure correct dtypes
        cls_targets = tf.cast(cls_targets, tf.int32)
        box_targets = tf.cast(box_targets, tf.float32)
        
        batch_size = tf.shape(cls_preds)[0]
        num_anchors = tf.shape(cls_preds)[1]
        
        # Positive anchors mask (class > 0)
        pos_mask = cls_targets > 0  # [batch, num_anchors]
        pos_mask_float = tf.cast(pos_mask, tf.float32)
        num_pos = tf.reduce_sum(pos_mask_float, axis=1)  # [batch]
        
        # === Classification Loss with Hard Negative Mining ===
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=cls_targets,
            logits=cls_preds
        )  # [batch, num_anchors]
        
        # Positive classification loss
        pos_cls_loss = cls_loss * pos_mask_float  # [batch, num_anchors]
        
        # Negative samples (background class = 0)
        neg_mask = tf.cast(cls_targets == 0, tf.float32)  # [batch, num_anchors]
        neg_cls_loss = cls_loss * neg_mask  # [batch, num_anchors]
        
        # Hard negative mining: select top-k negatives per sample
        num_neg = tf.minimum(
            tf.cast(num_pos * self.neg_pos_ratio, tf.int32),
            tf.reduce_sum(tf.cast(neg_mask > 0, tf.int32), axis=1)
        )  # [batch]
        num_neg = tf.maximum(num_neg, 1)  # At least 1 negative
        
        # Sort negative losses in descending order
        neg_cls_loss_sorted = tf.sort(neg_cls_loss, direction='DESCENDING', axis=1)
        
        # Create mask for top-k negatives
        indices = tf.range(num_anchors)
        neg_keep_mask = tf.cast(
            indices[tf.newaxis, :] < num_neg[:, tf.newaxis], 
            tf.float32
        )  # [batch, num_anchors]
        
        # Sum losses
        total_pos_cls_loss = tf.reduce_sum(pos_cls_loss)
        total_neg_cls_loss = tf.reduce_sum(neg_cls_loss_sorted * neg_keep_mask)
        
        total_num_pos = tf.maximum(tf.reduce_sum(num_pos), 1.0)
        cls_loss_normalized = (total_pos_cls_loss + total_neg_cls_loss) / total_num_pos
        
        # === Box Regression Loss ===
        # Only compute for positive anchors
        box_diff = self.smooth_l1_loss(box_targets, box_preds)  # [batch, num_anchors, 4]
        box_loss_per_anchor = tf.reduce_sum(box_diff, axis=-1)  # [batch, num_anchors]
        
        # Mask and sum box loss
        box_loss_masked = box_loss_per_anchor * pos_mask_float  # [batch, num_anchors]
        total_box_loss = tf.reduce_sum(box_loss_masked)
        box_loss_normalized = total_box_loss / total_num_pos
        
        return cls_loss_normalized + box_loss_normalized


# =============================================================================
# ANCHOR MATCHING
# =============================================================================

def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    xmin = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    ymin = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    xmax = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    ymax = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    inter_w = np.maximum(0, xmax - xmin)
    inter_h = np.maximum(0, ymax - ymin)
    intersection = inter_w * inter_h
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union = area1[:, None] + area2[None, :] - intersection
    
    return intersection / np.maximum(union, 1e-8)


def match_anchors_to_gt(anchors, gt_boxes, gt_labels, 
                        pos_iou_threshold=0.5, neg_iou_threshold=0.3):
    """Match anchors to ground truth boxes"""
    num_anchors = len(anchors)
    
    if len(gt_boxes) == 0:
        return np.zeros(num_anchors), np.zeros((num_anchors, 4))
    
    # Convert anchors to [xmin, ymin, xmax, ymax]
    anchor_boxes = np.zeros((num_anchors, 4))
    anchor_boxes[:, 0] = anchors[:, 0] - anchors[:, 2] / 2
    anchor_boxes[:, 1] = anchors[:, 1] - anchors[:, 3] / 2
    anchor_boxes[:, 2] = anchors[:, 0] + anchors[:, 2] / 2
    anchor_boxes[:, 3] = anchors[:, 1] + anchors[:, 3] / 2
    
    iou = compute_iou(anchor_boxes, gt_boxes)
    
    best_gt_idx = np.argmax(iou, axis=1)
    best_gt_iou = np.max(iou, axis=1)
    
    cls_targets = np.zeros(num_anchors, dtype=np.int32)
    box_targets = np.zeros((num_anchors, 4), dtype=np.float32)
    
    pos_mask = best_gt_iou >= pos_iou_threshold
    cls_targets[pos_mask] = gt_labels[best_gt_idx[pos_mask]]
    
    for gt_idx in range(len(gt_boxes)):
        best_anchor_idx = np.argmax(iou[:, gt_idx])
        cls_targets[best_anchor_idx] = gt_labels[gt_idx]
        pos_mask[best_anchor_idx] = True
    
    for i in np.where(pos_mask)[0]:
        gt_idx = best_gt_idx[i]
        gt_box = gt_boxes[gt_idx]
        
        gt_cx = (gt_box[0] + gt_box[2]) / 2
        gt_cy = (gt_box[1] + gt_box[3]) / 2
        gt_w = gt_box[2] - gt_box[0]
        gt_h = gt_box[3] - gt_box[1]
        
        box_targets[i, 0] = (gt_cx - anchors[i, 0]) / anchors[i, 2]
        box_targets[i, 1] = (gt_cy - anchors[i, 1]) / anchors[i, 3]
        box_targets[i, 2] = np.log(gt_w / anchors[i, 2] + 1e-8)
        box_targets[i, 3] = np.log(gt_h / anchors[i, 3] + 1e-8)
    
    return cls_targets, box_targets


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    model = SSDDetector(num_classes=6, input_size=300)
    
    sample_input = tf.random.normal((1, 300, 300, 3))
    cls_preds, box_preds = model(sample_input)
    
    print(f"Input: {sample_input.shape}")
    print(f"Class predictions: {cls_preds.shape}")
    print(f"Box predictions: {box_preds.shape}")
    print(f"Anchors: {model.num_anchors}")
    
    model.summary()
