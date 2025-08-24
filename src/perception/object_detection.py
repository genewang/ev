"""
Object detection module for EV autonomous trucking.

Handles 3D object detection from point cloud features using
advanced neural network architectures optimized for autonomous driving.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class ObjectDetector:
    """
    3D object detection system for autonomous driving.
    
    Features:
    - 3D bounding box detection
    - Multi-class object classification
    - Confidence scoring and NMS
    - Real-time detection pipeline
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """Initialize the object detector."""
        self.config = config
        self.device = device
        
        # Detection parameters
        self.num_classes = config.get('num_classes', 10)
        self.feature_dim = config.get('feature_dim', 1024)
        self.max_objects = config.get('max_objects', 100)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.7)
        
        # Object classes (typical for autonomous driving)
        self.object_classes = [
            'car', 'truck', 'bus', 'motorcycle', 'bicycle',
            'pedestrian', 'traffic_sign', 'traffic_light', 'construction', 'other'
        ]
        
        # Initialize detection network
        self._init_detection_network()
        
        # Performance tracking
        self.detection_times = []
        self.detection_counts = []
        
        logger.info(f"🎯 Object detector initialized: {self.num_classes} classes, {self.feature_dim} features")
        
    def _init_detection_network(self):
        """Initialize the object detection network."""
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Detection heads
        self.classification_head = nn.Linear(self.feature_dim // 4, self.num_classes)
        self.regression_head = nn.Linear(self.feature_dim // 4, 7)  # x, y, z, l, w, h, theta
        self.confidence_head = nn.Linear(self.feature_dim // 4, 1)
        
        # Anchor generation
        self.anchor_generator = self._create_anchors()
        
        # Move to device
        self.to(self.device)
        
    def _create_anchors(self) -> torch.Tensor:
        """Create anchor boxes for detection."""
        # Create a set of predefined anchor boxes
        # In production, this would be learned or carefully designed
        anchors = torch.tensor([
            [0, 0, 0, 4.0, 1.8, 1.5, 0],  # Car-like
            [0, 0, 0, 8.0, 2.5, 3.0, 0],  # Truck-like
            [0, 0, 0, 2.0, 0.8, 1.5, 0],  # Motorcycle-like
            [0, 0, 0, 0.8, 0.6, 1.8, 0],  # Pedestrian-like
        ], dtype=torch.float32)
        
        return anchors.to(self.device)
        
    def to(self, device):
        """Move the detector to the specified device."""
        self.device = device
        self.feature_processor.to(device)
        self.classification_head.to(device)
        self.regression_head.to(device)
        self.confidence_head.to(device)
        self.anchor_generator = self.anchor_generator.to(device)
        
    async def detect_objects(self, features: torch.Tensor, points: Optional[torch.Tensor] = None) -> Dict:
        """
        Detect objects from point cloud features.
        
        Args:
            features: Extracted features [batch_size, feature_dim]
            points: Optional point cloud data for visualization
            
        Returns:
            Dictionary containing detection results
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            logger.info(f"🎯 Detecting objects from {features.shape[0]} feature vectors...")
            
            # Move inputs to device
            features = features.to(self.device)
            if points is not None:
                points = points.to(self.device)
                
            # Process features
            processed_features = self.feature_processor(features)
            
            # Generate detections
            detections = await self._generate_detections(processed_features)
            
            # Apply NMS
            filtered_detections = await self._apply_nms(detections)
            
            # Post-process detections
            final_detections = await self._post_process_detections(filtered_detections, points)
            
            # Record detection time
            detection_time = asyncio.get_event_loop().time() - start_time
            self.detection_times.append(detection_time)
            self.detection_counts.append(len(final_detections))
            
            # Create result
            result = {
                'detections': final_detections,
                'raw_detections': detections,
                'detection_time': detection_time,
                'num_detections': len(final_detections),
                'confidence_scores': [det['confidence'] for det in final_detections],
                'object_classes': [det['class'] for det in final_detections]
            }
            
            logger.info(f"✅ Object detection completed: {len(final_detections)} objects in {detection_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in object detection: {e}")
            raise
            
    async def _generate_detections(self, features: torch.Tensor) -> List[Dict]:
        """Generate object detections from features."""
        logger.info("🔍 Generating object detections...")
        
        detections = []
        
        for i in range(features.shape[0]):
            feature = features[i]
            
            # Generate classification scores
            class_scores = F.softmax(self.classification_head(feature), dim=0)
            
            # Generate bounding box regression
            bbox_regression = self.regression_head(feature)
            
            # Generate confidence score
            confidence = torch.sigmoid(self.confidence_head(feature))
            
            # Find best class
            best_class_idx = torch.argmax(class_scores).item()
            best_class = self.object_classes[best_class_idx]
            best_score = class_scores[best_class_idx].item()
            
            # Create detection
            detection = {
                'bbox': bbox_regression.detach().cpu().numpy(),
                'class': best_class,
                'class_id': best_class_idx,
                'class_scores': class_scores.detach().cpu().numpy(),
                'confidence': confidence.item(),
                'feature_id': i
            }
            
            detections.append(detection)
            
        return detections
        
    async def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to detections."""
        logger.info("🔒 Applying NMS...")
        
        if len(detections) <= 1:
            return detections
            
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        
        for detection in sorted_detections:
            # Check if this detection overlaps significantly with already accepted ones
            should_add = True
            
            for accepted in filtered_detections:
                iou = self._compute_iou(detection['bbox'], accepted['bbox'])
                if iou > self.nms_threshold:
                    should_add = False
                    break
                    
            if should_add:
                filtered_detections.append(detection)
                
                # Limit maximum detections
                if len(filtered_detections) >= self.max_objects:
                    break
                    
        return filtered_detections
        
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        # Extract bbox parameters (x, y, z, l, w, h, theta)
        x1, y1, z1, l1, w1, h1, _ = bbox1
        x2, y2, z2, l2, w2, h2, _ = bbox2
        
        # For simplicity, compute 2D IoU (ignoring height and rotation)
        # In production, use proper 3D IoU computation
        
        # 2D bounding box corners
        x1_min, x1_max = x1 - l1/2, x1 + l1/2
        y1_min, y1_max = y1 - w1/2, y1 + w1/2
        
        x2_min, x2_max = x2 - l2/2, x2 + l2/2
        y2_min, y2_max = y2 - w2/2, y2 + w2/2
        
        # Compute intersection
        x_min = max(x1_min, x2_min)
        x_max = min(x1_max, x2_max)
        y_min = max(y1_min, y2_min)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
            
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Compute union
        area1 = l1 * w1
        area2 = l2 * w2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    async def _post_process_detections(self, detections: List[Dict], points: Optional[torch.Tensor]) -> List[Dict]:
        """Post-process detections for final output."""
        logger.info("🔧 Post-processing detections...")
        
        processed_detections = []
        
        for detection in detections:
            # Filter by confidence threshold
            if detection['confidence'] < self.confidence_threshold:
                continue
                
            # Process bounding box
            bbox = detection['bbox']
            processed_bbox = {
                'center': bbox[:3].tolist(),  # x, y, z
                'dimensions': bbox[3:6].tolist(),  # l, w, h
                'rotation': float(bbox[6])  # theta
            }
            
            # Create processed detection
            processed_detection = {
                'class': detection['class'],
                'class_id': detection['class_id'],
                'confidence': detection['confidence'],
                'bbox': processed_bbox,
                'class_scores': detection['class_scores'].tolist()
            }
            
            # Add point cloud association if available
            if points is not None:
                processed_detection['point_indices'] = self._get_points_in_bbox(
                    points, processed_bbox
                )
                
            processed_detections.append(processed_detection)
            
        return processed_detections
        
    def _get_points_in_bbox(self, points: torch.Tensor, bbox: Dict) -> List[int]:
        """Get indices of points inside the bounding box."""
        # Extract bbox parameters
        center = torch.tensor(bbox['center']).to(self.device)
        dimensions = torch.tensor(bbox['dimensions']).to(self.device)
        rotation = bbox['rotation']
        
        # Simple point-in-box check (ignoring rotation for now)
        # In production, use proper 3D point-in-box computation
        
        half_dims = dimensions / 2
        
        # Check if points are within bbox bounds
        in_x = (points[:, 0] >= center[0] - half_dims[0]) & (points[:, 0] <= center[0] + half_dims[0])
        in_y = (points[:, 1] >= center[1] - half_dims[1]) & (points[:, 1] <= center[1] + half_dims[1])
        in_z = (points[:, 2] >= center[2] - half_dims[2]) & (points[:, 2] <= center[2] + half_dims[2])
        
        # Points inside bbox
        inside_mask = in_x & in_y & in_z
        point_indices = torch.where(inside_mask)[0].tolist()
        
        return point_indices
        
    def get_detection_statistics(self) -> Dict:
        """Get object detection statistics."""
        if not self.detection_times:
            return {}
            
        stats = {
            'total_detections': len(self.detection_times),
            'avg_detection_time': np.mean(self.detection_times),
            'min_detection_time': np.min(self.detection_times),
            'max_detection_time': np.max(self.detection_times),
            'avg_objects_per_frame': np.mean(self.detection_counts),
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold
        }
        
        return stats
        
    def update_confidence_threshold(self, new_threshold: float):
        """Update the confidence threshold for detection."""
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        logger.info(f"🎯 Updated confidence threshold to {self.confidence_threshold}")
        
    def update_nms_threshold(self, new_threshold: float):
        """Update the NMS threshold for detection."""
        self.nms_threshold = max(0.0, min(1.0, new_threshold))
        logger.info(f"🔒 Updated NMS threshold to {self.nms_threshold}")
        
    def get_model_info(self) -> Dict:
        """Get information about the detection model."""
        model_info = {
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'max_objects': self.max_objects,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'object_classes': self.object_classes,
            'device': str(self.device)
        }
        
        return model_info
