"""
Feature extractor module for EV autonomous trucking.

Handles feature extraction from PointNet++ outputs and provides
interfaces for downstream perception tasks like object detection
and semantic segmentation.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .pointnet import PointNetPlusPlus


class FeatureExtractor:
    """
    Feature extraction system for autonomous driving perception.
    
    Features:
    - PointNet++ feature extraction
    - Multi-scale feature aggregation
    - Feature normalization and enhancement
    - Real-time feature processing
    """
    
    def __init__(self, model: PointNetPlusPlus, device: torch.device):
        """Initialize the feature extractor."""
        self.model = model
        self.device = device
        
        # Feature extraction parameters
        self.feature_dim = 1024  # PointNet++ output dimension
        self.num_scales = 3  # Multi-scale processing
        self.dropout_rate = 0.1
        
        # Initialize feature processing layers
        self._init_feature_layers()
        
        # Performance tracking
        self.extraction_times = []
        self.feature_cache = {}
        
        logger.info(f"🔍 Feature extractor initialized on {device}")
        
    def _init_feature_layers(self):
        """Initialize feature processing layers."""
        # Feature normalization
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        
        # Multi-scale feature processing
        self.scale_conv1 = nn.Conv1d(self.feature_dim, self.feature_dim, 1)
        self.scale_conv2 = nn.Conv1d(self.feature_dim, self.feature_dim, 1)
        self.scale_conv3 = nn.Conv1d(self.feature_dim, self.feature_dim, 1)
        
        # Feature enhancement
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Linear(self.feature_dim * self.num_scales, self.feature_dim)
        
        # Move to device
        self.to(self.device)
        
    def to(self, device):
        """Move the feature extractor to the specified device."""
        self.device = device
        self.feature_norm.to(device)
        self.scale_conv1.to(device)
        self.scale_conv2.to(device)
        self.scale_conv3.to(device)
        self.feature_enhancer.to(device)
        self.feature_fusion.to(device)
        
    async def extract_features(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> Dict:
        """
        Extract features from point cloud data.
        
        Args:
            points: Point cloud data [batch_size, num_points, 3]
            features: Optional point features [batch_size, num_points, feature_dim]
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            logger.info(f"🔍 Extracting features from {points.shape[1]} points...")
            
            # Move inputs to device
            points = points.to(self.device)
            if features is not None:
                features = features.to(self.device)
                
            # Extract features using PointNet++
            with torch.no_grad():
                pointnet_features = self.model(points, features)
                
            # Process features at multiple scales
            multi_scale_features = await self._extract_multi_scale_features(pointnet_features)
            
            # Enhance features
            enhanced_features = await self._enhance_features(multi_scale_features)
            
            # Compute feature statistics
            feature_stats = self._compute_feature_statistics(enhanced_features)
            
            # Cache features for reuse
            self._cache_features(enhanced_features, feature_stats)
            
            # Record extraction time
            extraction_time = asyncio.get_event_loop().time() - start_time
            self.extraction_times.append(extraction_time)
            
            # Create result
            result = {
                'features': enhanced_features,
                'multi_scale_features': multi_scale_features,
                'pointnet_features': pointnet_features,
                'feature_stats': feature_stats,
                'extraction_time': extraction_time,
                'num_points': points.shape[1],
                'feature_dim': enhanced_features.shape[-1]
            }
            
            logger.info(f"✅ Feature extraction completed in {extraction_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in feature extraction: {e}")
            raise
            
    async def _extract_multi_scale_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract features at multiple scales."""
        logger.info("📏 Extracting multi-scale features...")
        
        # Apply different scale convolutions
        scale1_features = F.relu(self.scale_conv1(features.transpose(1, 2)).transpose(1, 2))
        scale2_features = F.relu(self.scale_conv2(features.transpose(1, 2)).transpose(1, 2))
        scale3_features = F.relu(self.scale_conv3(features.transpose(1, 2)).transpose(1, 2))
        
        # Global pooling for each scale
        global_scale1 = torch.mean(scale1_features, dim=1)  # [batch_size, feature_dim]
        global_scale2 = torch.mean(scale2_features, dim=1)  # [batch_size, feature_dim]
        global_scale3 = torch.mean(scale3_features, dim=1)  # [batch_size, feature_dim]
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([global_scale1, global_scale2, global_scale3], dim=-1)
        
        return multi_scale_features
        
    async def _enhance_features(self, multi_scale_features: torch.Tensor) -> torch.Tensor:
        """Enhance features using neural networks."""
        logger.info("✨ Enhancing features...")
        
        # Apply feature enhancement
        enhanced_features = self.feature_enhancer(multi_scale_features)
        
        # Apply feature fusion
        fused_features = self.feature_fusion(multi_scale_features)
        
        # Combine enhanced and fused features
        final_features = enhanced_features + fused_features
        
        # Apply final normalization
        final_features = self.feature_norm(final_features)
        
        return final_features
        
    def _compute_feature_statistics(self, features: torch.Tensor) -> Dict:
        """Compute statistics for extracted features."""
        with torch.no_grad():
            stats = {
                'mean': float(torch.mean(features)),
                'std': float(torch.std(features)),
                'min': float(torch.min(features)),
                'max': float(torch.max(features)),
                'feature_magnitude': float(torch.norm(features)),
                'feature_sparsity': float(torch.sum(features == 0) / features.numel())
            }
            
        return stats
        
    def _cache_features(self, features: torch.Tensor, stats: Dict):
        """Cache features for potential reuse."""
        cache_key = f"features_{len(self.feature_cache)}"
        self.feature_cache[cache_key] = {
            'features': features.clone(),
            'stats': stats,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        # Limit cache size
        if len(self.feature_cache) > 100:
            # Remove oldest entry
            oldest_key = min(self.feature_cache.keys(), key=lambda k: self.feature_cache[k]['timestamp'])
            del self.feature_cache[oldest_key]
            
    async def get_cached_features(self, cache_key: str) -> Optional[Dict]:
        """Retrieve cached features."""
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        return None
        
    def get_extraction_statistics(self) -> Dict:
        """Get feature extraction statistics."""
        if not self.extraction_times:
            return {}
            
        stats = {
            'total_extractions': len(self.extraction_times),
            'avg_extraction_time': np.mean(self.extraction_times),
            'min_extraction_time': np.min(self.extraction_times),
            'max_extraction_time': np.max(self.extraction_times),
            'std_extraction_time': np.std(self.extraction_times),
            'cache_size': len(self.feature_cache)
        }
        
        return stats
        
    def clear_cache(self):
        """Clear the feature cache."""
        self.feature_cache.clear()
        logger.info("🗑️ Feature cache cleared")
        
    def get_model_info(self) -> Dict:
        """Get information about the underlying model."""
        model_info = {
            'model_type': type(self.model).__name__,
            'feature_dim': self.feature_dim,
            'num_scales': self.num_scales,
            'dropout_rate': self.dropout_rate,
            'device': str(self.device)
        }
        
        return model_info
