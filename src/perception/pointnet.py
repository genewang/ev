"""
Full PointNet++ encoder for EV autonomous trucking perception.

Three set-abstraction levels, farthest-point sampling, ball-query grouping,
and local MLPs. The production schedule is 512 points, then 128, then a
global set. Neural Engine and TensorRT are the deployment targets for the
32 ms vehicle budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from loguru import logger


class PointNetPlusPlus(nn.Module):
    """
    Full PointNet++ for the vehicle perception stack.

    Level 1 samples 512 points at radius 0.2 with 32 neighbors.
    Level 2 samples 128 points at radius 0.4 with 64 neighbors.
    Level 3 pools the whole set into a 1024-d global feature, then classifies.
    """
    
    def __init__(self, num_classes: int = 20, num_features: int = 64, device: torch.device = None):
        """
        Initialize PointNet++.
        
        Args:
            num_classes: Number of semantic classes
            num_features: Number of input features per point
            device: Target device for computation
        """
        super(PointNetPlusPlus, self).__init__()
        
        self.num_classes = num_classes
        self.num_features = num_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Full PointNet++ schedule. in_channel includes the 3 relative XYZ
        # coordinates that each set-abstraction layer concatenates.
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=num_features + 3, mlp=[64, 64, 128],
            device=self.device
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256],
            device=self.device
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=256 + 3, mlp=[256, 512, 1024], 
            device=self.device
        )
        
        # Feature transformation layers
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, num_classes)
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"🚀 PointNet++ initialized on {self.device}")
        
    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PointNet++.
        
        Args:
            xyz: Point coordinates (B, N, 3)
            features: Point features (B, N, C)
            
        Returns:
            Global features (B, 1024)
        """
        B, _, _ = xyz.shape
        
        # Set abstraction layers
        l1_xyz, l1_features = self.sa1(xyz, features)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        
        # Global feature extraction
        x = l3_features.view(B, 1024)
        
        # Classification head
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x
        
    def extract_features(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features for downstream tasks.
        
        Args:
            xyz: Point coordinates (B, N, 3)
            features: Point features (B, N, C)
            
        Returns:
            Multi-scale features dictionary
        """
        B, _, _ = xyz.shape
        
        # Extract features at each level
        l1_xyz, l1_features = self.sa1(xyz, features)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        
        return {
            'level1': {'xyz': l1_xyz, 'features': l1_features},
            'level2': {'xyz': l2_xyz, 'features': l2_features},
            'level3': {'xyz': l3_xyz, 'features': l3_features}
        }


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer.
    
    Implements efficient point sampling and grouping with MLP feature extraction.
    """
    
    def __init__(self, npoint: Optional[int], radius: Optional[float], 
                 nsample: Optional[int], in_channel: int, mlp: list, device: torch.device):
        """
        Initialize Set Abstraction layer.
        
        Args:
            npoint: Number of points to sample (None for global pooling)
            radius: Ball query radius
            nsample: Maximum number of points in ball query
            in_channel: Number of input channels
            mlp: MLP layer dimensions
            device: Target device
        """
        super(PointNetSetAbstraction, self).__init__()
        
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.device = device
        
        # MLP layers
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            
        self.to(device)
        
    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Set Abstraction layer.
        
        Args:
            xyz: Point coordinates (B, N, 3)
            points: Point features (B, N, C)
            
        Returns:
            Tuple of (new_xyz, new_points)
        """
        xyz = xyz.contiguous()
        if points is not None:
            points = points.contiguous()

        if self.npoint is not None:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_points = torch.cat(
                    [grouped_xyz, index_points(points, idx)], dim=-1
                )
            else:
                grouped_points = grouped_xyz
        else:
            # One group containing every point, centered on the cloud centroid.
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.unsqueeze(1) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_points = torch.cat(
                    [grouped_xyz, points.unsqueeze(1)], dim=-1
                )
            else:
                grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 2, 1).contiguous()
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
            
        # Max pooling
        new_points = torch.max(grouped_points, 2)[0]  # (B, C, npoint)
        
        return new_xyz, new_points.permute(0, 2, 1)


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) for efficient point selection.
    
    Args:
        xyz: Point coordinates (B, N, 3)
        npoint: Number of points to sample
        
    Returns:
        Indices of sampled points (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Randomly select first point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index points using given indices.
    
    Args:
        points: Input points (B, N, C)
        idx: Indices (B, S)
        
    Returns:
        Indexed points (B, S, C)
    """
    device = points.device
    B = points.shape[0]
    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    
    new_points = points[batch_indices, idx, :]
    return new_points


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """
    Group source points around each query point.

    Neighbors outside ``radius`` are ignored. If a query has fewer than
    ``nsample`` neighbors, the nearest in-radius point is repeated. If the
    ball is empty, the nearest source point is used so the index tensor stays
    valid.

    Args:
        radius: Query radius
        nsample: Neighbors returned per query
        xyz: Source points (B, N, 3)
        query: Query points (B, S, 3)

    Returns:
        Group indices (B, S, nsample)
    """
    dist = torch.cdist(query, xyz)
    outside = dist > radius
    masked = dist.masked_fill(outside, float("inf"))
    empty = torch.isinf(masked).all(dim=-1, keepdim=True)
    masked = torch.where(empty, dist, masked)

    k = min(nsample, xyz.shape[1])
    idx = torch.topk(masked, k=k, dim=-1, largest=False).indices
    if k < nsample:
        pad = idx[:, :, :1].expand(-1, -1, nsample - k)
        idx = torch.cat([idx, pad], dim=-1)
    return idx


class NeuralEngineOptimizer:
    """
    Neural Engine and inference optimization for the full PointNet++.

    Mixed precision and TorchScript are the development-host path. The vehicle
    GPU path builds a TensorRT engine from the same module. Apple Silicon uses
    Core ML. See ``src/inference/targets.py``.
    """
    
    @staticmethod
    def optimize_model(model: nn.Module) -> nn.Module:
        """
        Apply Neural Engine optimizations to the model.
        
        Args:
            model: Input model
            
        Returns:
            Optimized model
        """
        # Enable mixed precision for better performance
        model = model.half()
        
        # Enable JIT compilation if available
        try:
            model = torch.jit.script(model)
            logger.info("✅ JIT compilation applied")
        except Exception as e:
            logger.warning(f"⚠️ JIT compilation failed: {e}")
            
        return model
        
    @staticmethod
    def optimize_inference(model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """
        Optimize model for inference with sample input.
        
        Args:
            model: Input model
            sample_input: Sample input for optimization
            
        Returns:
            Optimized model
        """
        # Enable inference optimizations
        model.eval()
        
        # Warm up with sample input
        with torch.no_grad():
            _ = model(*sample_input)
            
        return model
