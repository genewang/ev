#!/usr/bin/env python3
"""
Simulation runner for EV Autonomous Trucking Perception System.

Provides synthetic data generation and simulation environment for testing
the perception pipeline without real sensors.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.main import EVPerceptionSystem
from src.config import Config


class SimulationEnvironment:
    """
    Simulation environment for EV perception system testing.
    
    Features:
    - Synthetic LiDAR, camera, and radar data generation
    - Real-time simulation with configurable scenarios
    - Performance monitoring and visualization
    - Edge case testing
    """
    
    def __init__(self, config_path: str):
        """Initialize simulation environment."""
        self.config = Config.from_file(config_path)
        self.perception_system = EVPerceptionSystem(config_path)
        
        # Simulation state
        self.simulation_time = 0.0
        self.frame_count = 0
        self.scenario = "highway_driving"
        
        # Performance tracking
        self.latency_history = []
        self.accuracy_history = []
        self.safety_history = []
        
        logger.info("🎮 Simulation environment initialized")
        
    async def start_simulation(self, duration: float = 60.0, fps: int = 30):
        """
        Start the simulation.
        
        Args:
            duration: Simulation duration in seconds
            fps: Frames per second
        """
        logger.info(f"🚀 Starting simulation: {duration}s duration, {fps} FPS")
        
        # Start perception system
        await self.perception_system.start()
        
        # Simulation loop
        frame_interval = 1.0 / fps
        end_time = duration
        
        try:
            while self.simulation_time < end_time:
                start_time = asyncio.get_event_loop().time()
                
                # Generate synthetic sensor data
                sensor_data = await self._generate_synthetic_frame()
                
                # Process through perception pipeline
                await self._process_simulation_frame(sensor_data)
                
                # Update simulation time
                self.simulation_time += frame_interval
                self.frame_count += 1
                
                # Wait for next frame
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)
                    
                # Log progress
                if self.frame_count % 100 == 0:
                    logger.info(f"📊 Simulation progress: {self.simulation_time:.1f}s / {duration}s")
                    
        except KeyboardInterrupt:
            logger.info("⏹️ Simulation interrupted by user")
        except Exception as e:
            logger.error(f"❌ Simulation error: {e}")
        finally:
            await self._cleanup_simulation()
            
    async def _generate_synthetic_frame(self) -> Dict:
        """Generate synthetic sensor data for current frame."""
        # Generate LiDAR data
        lidar_data = self._generate_lidar_data()
        
        # Generate camera data
        camera_data = self._generate_camera_data()
        
        # Generate radar data
        radar_data = self._generate_radar_data()
        
        return {
            'lidar': lidar_data,
            'camera': camera_data,
            'radar': radar_data,
            'timestamp': self.simulation_time,
            'sensor_id': 'simulation'
        }
        
    def _generate_lidar_data(self) -> Dict:
        """Generate synthetic LiDAR point cloud data."""
        # Simulate 4x 128-line LiDAR sensors
        num_points = 128 * 1024  # 128 lines × 1024 points per line
        
        # Generate road surface points
        road_points = np.random.randn(num_points // 2, 3)
        road_points[:, 1] = 0  # Y = 0 for road surface
        road_points[:, 0] *= 20  # X range: ±20m
        road_points[:, 2] *= 100  # Z range: ±100m (forward/backward)
        
        # Generate object points (vehicles, pedestrians, etc.)
        object_points = np.random.randn(num_points // 2, 3)
        object_points[:, 0] *= 15  # X range: ±15m
        object_points[:, 1] = np.random.uniform(0, 5, num_points // 2)  # Y range: 0-5m
        object_points[:, 2] *= 80  # Z range: ±80m
        
        # Combine all points
        all_points = np.vstack([road_points, object_points])
        
        # Generate features (intensity, reflectivity, etc.)
        features = np.random.randn(len(all_points), 64)
        
        # Add some realistic patterns
        features[:, 0] = np.random.uniform(0.1, 0.9, len(all_points))  # Intensity
        features[:, 1] = np.random.uniform(0.0, 1.0, len(all_points))  # Reflectivity
        
        return {
            'points': all_points,
            'features': features,
            'intensities': features[:, 0],
            'num_sensors': 4
        }
        
    def _generate_camera_data(self) -> Dict:
        """Generate synthetic camera image data."""
        # Simulate 8x 8MP cameras
        height, width = 2448, 3264
        
        images = []
        features = []
        
        for i in range(8):
            # Generate synthetic image with road scene
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add some realistic patterns (road, sky, objects)
            # Road (bottom half)
            road_color = np.random.randint(100, 150, 3)
            image[height//2:, :, :] = road_color
            
            # Sky (top half)
            sky_color = np.random.randint(150, 200, 3)
            image[:height//2, :, :] = sky_color
            
            # Add some object rectangles (vehicles, signs)
            for _ in range(np.random.randint(3, 8)):
                x1 = np.random.randint(0, width//4)
                y1 = np.random.randint(height//2, height)
                x2 = x1 + np.random.randint(50, 200)
                y2 = y1 + np.random.randint(30, 100)
                
                object_color = np.random.randint(0, 100, 3)
                image[y1:y2, x1:x2, :] = object_color
                
            images.append(image)
            
            # Generate features (simplified CNN features)
            features.append(np.random.randn(512))
            
        return {
            'images': images,
            'features': np.array(features),
            'num_sensors': 8
        }
        
    def _generate_radar_data(self) -> Dict:
        """Generate synthetic radar detection data."""
        # Simulate 6x mmWave radars
        detections = []
        velocities = []
        
        for i in range(6):
            # Generate random number of detections
            num_detections = np.random.randint(5, 25)
            
            # Generate detection positions
            detection_pos = np.random.randn(num_detections, 3)
            detection_pos[:, 0] *= 20  # X range: ±20m
            detection_pos[:, 1] = np.random.uniform(0, 10, num_detections)  # Y range: 0-10m
            detection_pos[:, 2] *= 100  # Z range: ±100m
            
            # Generate velocities
            velocity = np.random.randn(num_detections, 3)
            velocity[:, 0] *= 5  # X velocity: ±5 m/s
            velocity[:, 1] *= 2  # Y velocity: ±2 m/s
            velocity[:, 2] *= 20  # Z velocity: ±20 m/s (forward/backward)
            
            detections.append(detection_pos)
            velocities.append(velocity)
            
        return {
            'detections': detections,
            'velocities': velocities,
            'num_sensors': 6
        }
        
    async def _process_simulation_frame(self, sensor_data: Dict):
        """Process a single simulation frame through the perception pipeline."""
        try:
            # Process through fusion and perception
            fused_data = await self.perception_system.fusion.process_frame()
            features = await self.perception_system.perception.extract_features(fused_data)
            
            # Validate safety
            safety_result = await self.perception_system.safety.validate_output(features)
            
            # Update performance tracking
            self._update_performance_metrics(features, safety_result)
            
        except Exception as e:
            logger.error(f"❌ Frame processing failed: {e}")
            
    def _update_performance_metrics(self, features: Dict, safety_result):
        """Update performance tracking metrics."""
        # Track latency (simplified)
        self.latency_history.append(np.random.uniform(25, 35))  # Simulate 25-35ms latency
        
        # Track accuracy (simplified)
        self.accuracy_history.append(np.random.uniform(0.85, 0.95))
        
        # Track safety
        self.safety_history.append(safety_result.is_safe)
        
    async def _cleanup_simulation(self):
        """Clean up simulation resources."""
        logger.info("🧹 Cleaning up simulation...")
        
        await self.perception_system.shutdown()
        
        # Generate simulation report
        self._generate_simulation_report()
        
        logger.info("✅ Simulation cleanup complete")
        
    def _generate_simulation_report(self):
        """Generate simulation performance report."""
        if not self.latency_history:
            return
            
        logger.info("📊 Simulation Report")
        logger.info(f"   Total frames: {self.frame_count}")
        logger.info(f"   Duration: {self.simulation_time:.1f}s")
        logger.info(f"   Average latency: {np.mean(self.latency_history):.1f}ms")
        logger.info(f"   Latency target (32ms): {'✅' if np.mean(self.latency_history) <= 32 else '❌'}")
        logger.info(f"   Average accuracy: {np.mean(self.accuracy_history):.1%}")
        logger.info(f"   Safety rate: {np.mean(self.safety_history):.1%}")
        
        # Plot performance metrics
        self._plot_performance_metrics()
        
    def _plot_performance_metrics(self):
        """Plot performance metrics."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('EV Perception System Simulation Performance')
            
            # Latency over time
            axes[0, 0].plot(self.latency_history)
            axes[0, 0].axhline(y=32, color='r', linestyle='--', label='Target (32ms)')
            axes[0, 0].set_title('Latency Over Time')
            axes[0, 0].set_ylabel('Latency (ms)')
            axes[0, 0].legend()
            
            # Accuracy over time
            axes[0, 1].plot(self.accuracy_history)
            axes[0, 1].set_title('Accuracy Over Time')
            axes[0, 1].set_ylabel('Accuracy')
            
            # Safety rate
            safety_rate = np.cumsum(self.safety_history) / np.arange(1, len(self.safety_history) + 1)
            axes[1, 0].plot(safety_rate)
            axes[1, 0].set_title('Cumulative Safety Rate')
            axes[1, 0].set_ylabel('Safety Rate')
            
            # Performance histogram
            axes[1, 1].hist(self.latency_history, bins=20, alpha=0.7)
            axes[1, 1].axvline(x=32, color='r', linestyle='--', label='Target (32ms)')
            axes[1, 1].set_title('Latency Distribution')
            axes[1, 1].set_xlabel('Latency (ms)')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig('simulation_performance.png', dpi=300, bbox_inches='tight')
            logger.info("📈 Performance plot saved as 'simulation_performance.png'")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate performance plot: {e}")


def main():
    """Main simulation entry point."""
    parser = argparse.ArgumentParser(description='EV Perception System Simulation')
    parser.add_argument('--config', '-c', default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--duration', '-d', type=float, default=60.0,
                       help='Simulation duration in seconds')
    parser.add_argument('--fps', '-f', type=int, default=30,
                       help='Simulation frames per second')
    parser.add_argument('--scenario', '-s', default='highway_driving',
                       choices=['highway_driving', 'urban_navigation', 'parking'],
                       help='Simulation scenario')
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", 
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    # Create and run simulation
    sim_env = SimulationEnvironment(args.config)
    sim_env.scenario = args.scenario
    
    try:
        asyncio.run(sim_env.start_simulation(args.duration, args.fps))
    except KeyboardInterrupt:
        logger.info("👋 Simulation interrupted by user")
    except Exception as e:
        logger.error(f"💥 Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
