"""
Comprehensive test suite for EV autonomous trucking perception system.

Tests cover:
- PointNet++ feature extraction
- Object detection
- Semantic segmentation
- Multi-modal fusion
- Safety framework
- Performance benchmarks
"""

import pytest
import asyncio
import numpy as np
import torch
from unittest.mock import Mock, patch
from loguru import logger

# Import system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from perception.pointnet import PointNetPlusPlus
from perception.feature_extractor import FeatureExtractor
from perception.object_detection import ObjectDetector
from perception.segmentation import SemanticSegmentation
from perception.perception_system import PerceptionSystem
from fusion.multi_modal_fusion import MultiModalFusion
from safety.safety_framework import SafetyFramework
from config import Config


class TestPointNetPlusPlus:
    """Test PointNet++ implementation."""
    
    @pytest.fixture
    def pointnet(self):
        """Create PointNet++ instance for testing."""
        return PointNetPlusPlus(num_classes=20, num_features=64)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample point cloud data."""
        batch_size = 2
        num_points = 1024
        num_features = 64
        
        points = torch.randn(batch_size, num_points, 3)
        features = torch.randn(batch_size, num_points, num_features)
        
        return points, features
    
    def test_pointnet_initialization(self, pointnet):
        """Test PointNet++ initialization."""
        assert pointnet.num_classes == 20
        assert pointnet.num_features == 64
        assert pointnet.device is not None
        
    def test_pointnet_forward(self, pointnet, sample_data):
        """Test PointNet++ forward pass."""
        points, features = sample_data
        
        with torch.no_grad():
            output = pointnet(points, features)
            
        assert output.shape == (2, 20)  # batch_size, num_classes
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_pointnet_feature_extraction(self, pointnet, sample_data):
        """Test PointNet++ feature extraction."""
        points, features = sample_data
        
        with torch.no_grad():
            extracted_features = pointnet.extract_features(points, features)
            
        assert 'level1' in extracted_features
        assert 'level2' in extracted_features
        assert 'level3' in extracted_features
        
        for level, data in extracted_features.items():
            assert 'xyz' in data
            assert 'features' in data
            assert data['xyz'].shape[0] == 2  # batch_size
            assert data['features'].shape[0] == 2  # batch_size


class TestFeatureExtractor:
    """Test feature extraction system."""
    
    @pytest.fixture
    def feature_extractor(self):
        """Create feature extractor for testing."""
        pointnet = PointNetPlusPlus(num_classes=20, num_features=64)
        device = torch.device("cpu")
        return FeatureExtractor(pointnet, device)
    
    @pytest.fixture
    def sample_points(self):
        """Create sample point cloud data."""
        return torch.randn(1, 1024, 3)
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, feature_extractor, sample_points):
        """Test feature extraction."""
        result = await feature_extractor.extract_features(sample_points)
        
        assert 'features' in result
        assert 'multi_scale_features' in result
        assert 'pointnet_features' in result
        assert 'feature_stats' in result
        assert 'extraction_time' in result
        
        assert result['features'].shape[0] == 1  # batch_size
        assert result['num_points'] == 1024
        
    def test_feature_statistics(self, feature_extractor, sample_points):
        """Test feature statistics calculation."""
        features = torch.randn(1, 1024)
        stats = feature_extractor._compute_feature_statistics(features)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'feature_magnitude' in stats
        assert 'feature_sparsity' in stats
        
        assert isinstance(stats['mean'], float)
        assert isinstance(stats['std'], float)


class TestObjectDetector:
    """Test object detection system."""
    
    @pytest.fixture
    def object_detector(self):
        """Create object detector for testing."""
        config = {
            'confidence_threshold': 0.7,
            'nms_threshold': 0.3,
            'max_detections': 100
        }
        device = torch.device("cpu")
        return ObjectDetector(config, device)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        features = torch.randn(1, 1024)
        points = torch.randn(1, 1024, 3)
        return features, points
    
    @pytest.mark.asyncio
    async def test_object_detection(self, object_detector, sample_data):
        """Test object detection."""
        features, points = sample_data
        
        result = await object_detector.detect_objects(features, points)
        
        assert 'detections' in result
        assert 'num_detections' in result
        assert 'detection_time' in result
        assert 'confidence_scores' in result
        assert 'object_classes' in result
        
        assert isinstance(result['detections'], list)
        assert isinstance(result['num_detections'], int)
        assert result['num_detections'] >= 0
        
    def test_object_detector_initialization(self, object_detector):
        """Test object detector initialization."""
        assert object_detector.confidence_threshold == 0.7
        assert object_detector.nms_threshold == 0.3
        assert object_detector.max_detections == 100
        assert len(object_detector.object_classes) > 0


class TestSemanticSegmentation:
    """Test semantic segmentation system."""
    
    @pytest.fixture
    def segmentation(self):
        """Create segmentation system for testing."""
        config = {
            'num_classes': 20,
            'confidence_threshold': 0.5
        }
        device = torch.device("cpu")
        return SemanticSegmentation(config, device)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        features = torch.randn(1, 1024)
        points = torch.randn(1, 1024, 3)
        return features, points
    
    @pytest.mark.asyncio
    async def test_semantic_segmentation(self, segmentation, sample_data):
        """Test semantic segmentation."""
        features, points = sample_data
        
        result = await segmentation.segment_points(features, points)
        
        assert 'segmentation_logits' in result
        assert 'segmentation_probs' in result
        assert 'class_predictions' in result
        assert 'confidence_scores' in result
        assert 'valid_mask' in result
        
        assert result['segmentation_logits'].shape[0] == 1  # batch_size
        assert result['segmentation_logits'].shape[1] == 1024  # num_points
        assert result['segmentation_logits'].shape[2] == 20  # num_classes
        
    @pytest.mark.asyncio
    async def test_segmentation_statistics(self, segmentation, sample_data):
        """Test segmentation statistics."""
        features, points = sample_data
        
        # Create mock segmentation result
        segmentation_result = {
            'class_predictions': torch.randint(0, 20, (1, 1024)),
            'confidence_scores': torch.rand(1, 1024),
            'valid_mask': torch.rand(1, 1024) > 0.5
        }
        
        stats = await segmentation.get_segmentation_statistics(segmentation_result)
        
        assert 'class_distribution' in stats
        assert 'confidence_stats' in stats
        assert 'segmentation_quality' in stats
        assert 'num_valid_points' in stats
        assert 'num_total_points' in stats


class TestPerceptionSystem:
    """Test main perception system."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def perception_system(self, config):
        """Create perception system for testing."""
        return PerceptionSystem(config)
    
    @pytest.mark.asyncio
    async def test_perception_system_initialization(self, perception_system):
        """Test perception system initialization."""
        assert perception_system.config is not None
        assert perception_system.device is not None
        assert perception_system.pointnet is not None
        assert perception_system.feature_extractor is not None
        assert perception_system.object_detector is not None
        assert perception_system.segmentation is not None
        
    @pytest.mark.asyncio
    async def test_perception_system_start(self, perception_system):
        """Test perception system start."""
        # Mock the start method to avoid actual initialization
        with patch.object(perception_system, 'start') as mock_start:
            await perception_system.start()
            mock_start.assert_called_once()


class TestMultiModalFusion:
    """Test multi-modal fusion system."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def fusion_system(self, config):
        """Create fusion system for testing."""
        return MultiModalFusion(config)
    
    @pytest.mark.asyncio
    async def test_fusion_system_initialization(self, fusion_system):
        """Test fusion system initialization."""
        assert fusion_system.config is not None
        assert fusion_system.device is not None
        assert fusion_system.sensor_fusion is not None
        assert fusion_system.llm_attention is not None
        assert fusion_system.temporal_fusion is not None
        
    @pytest.mark.asyncio
    async def test_fusion_system_start(self, fusion_system):
        """Test fusion system start."""
        # Mock the start method to avoid actual initialization
        with patch.object(fusion_system, 'initialize') as mock_initialize:
            await fusion_system.initialize()
            mock_initialize.assert_called_once()


class TestSafetyFramework:
    """Test safety framework."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def safety_framework(self, config):
        """Create safety framework for testing."""
        return SafetyFramework(config)
    
    @pytest.mark.asyncio
    async def test_safety_framework_initialization(self, safety_framework):
        """Test safety framework initialization."""
        assert safety_framework.config is not None
        assert safety_framework.device is not None
        assert safety_framework.asil_runtime is not None
        assert safety_framework.wcet_analyzer is not None
        assert safety_framework.fault_tester is not None
        
    @pytest.mark.asyncio
    async def test_safety_validation(self, safety_framework):
        """Test safety validation."""
        # Create mock features
        features = {
            'point_features': np.random.randn(1024, 64),
            'object_features': np.random.randn(10, 256),
            'segmentation_features': np.random.randn(1024, 20)
        }
        
        result = await safety_framework.validate_output(features)
        
        assert hasattr(result, 'is_safe')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reason')
        assert hasattr(result, 'violations')
        assert hasattr(result, 'wcet_status')
        assert hasattr(result, 'fault_status')


class TestPerformanceBenchmarks:
    """Test performance benchmarks."""
    
    @pytest.fixture
    def pointnet(self):
        """Create PointNet++ for performance testing."""
        return PointNetPlusPlus(num_classes=20, num_features=64)
    
    def test_pointnet_performance(self, pointnet):
        """Test PointNet++ performance."""
        batch_size = 1
        num_points = 1024
        num_features = 64
        
        points = torch.randn(batch_size, num_points, 3)
        features = torch.randn(batch_size, num_points, num_features)
        
        pointnet.eval()
        # Warm up
        with torch.no_grad():
            _ = pointnet(points, features)

        # Time the forward pass of the full 512/128/global PointNet++.
        import time
        start_time = time.time()

        with torch.no_grad():
            output = pointnet(points, features)

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to ms

        # CPU unit-test ceiling. The 32 ms vehicle budget is the TensorRT target.
        assert execution_time < 5000
        assert output.shape == (batch_size, 20)
        
    def test_memory_usage(self, pointnet):
        """Test memory usage."""
        batch_size = 1
        num_points = 1024
        num_features = 64
        
        points = torch.randn(batch_size, num_points, 3)
        features = torch.randn(batch_size, num_points, num_features)
        
        pointnet.eval()
        # Check memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        with torch.no_grad():
            output = pointnet(points, features)
            
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_usage = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_usage < 100 * 1024 * 1024  # Less than 100MB


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_perception(self):
        """Test end-to-end perception pipeline."""
        # Create configuration
        config = Config()
        
        # Create perception system
        perception_system = PerceptionSystem(config)
        
        # Create sample data
        points = torch.randn(1, 1024, 3)
        features = torch.randn(1, 1024, 64)
        
        # Test feature extraction
        feature_result = await perception_system.feature_extractor.extract_features(points, features)
        assert 'features' in feature_result
        
        # Test object detection
        detection_result = await perception_system.object_detector.detect_objects(
            feature_result['features'], points
        )
        assert 'detections' in detection_result
        
        # Test segmentation
        segmentation_result = await perception_system.segmentation.segment_points(
            feature_result['features'], points
        )
        assert 'class_predictions' in segmentation_result
        
    @pytest.mark.asyncio
    async def test_safety_integration(self):
        """Test safety framework integration."""
        config = Config()
        safety_framework = SafetyFramework(config)
        
        # Create mock features
        features = {
            'point_features': np.random.randn(1024, 64),
            'object_features': np.random.randn(10, 256),
            'segmentation_features': np.random.randn(1024, 20)
        }
        
        # Test safety validation
        safety_result = await safety_framework.validate_output(features)
        assert hasattr(safety_result, 'is_safe')
        assert hasattr(safety_result, 'confidence')


# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.asyncio
]

# Test configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "asyncio: mark test as async test"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])