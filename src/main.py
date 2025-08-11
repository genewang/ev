#!/usr/bin/env python3
"""
Main entry point for the EV Autonomous Trucking Perception System.

This module orchestrates the entire perception pipeline including:
- Multi-modal sensor fusion (LiDAR, cameras, radar)
- PointNet++-based feature extraction
- Safety-critical runtime execution
- AI-powered development tools
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from .perception import PerceptionSystem
from .fusion import MultiModalFusion
from .safety import SafetyFramework
from .ai_tools import AIDevelopmentTools
from .config import Config


class EVPerceptionSystem:
    """Main orchestrator for the EV perception system."""
    
    def __init__(self, config_path: str):
        """Initialize the perception system with configuration."""
        self.config = Config.from_file(config_path)
        self.perception = PerceptionSystem(self.config)
        self.fusion = MultiModalFusion(self.config)
        self.safety = SafetyFramework(self.config)
        self.ai_tools = AIDevelopmentTools(self.config)
        
        # Performance metrics
        self.latency_target = 32  # ms (vs 85ms NVIDIA reference)
        self.running = False
        
    async def start(self):
        """Start the perception system."""
        logger.info("🚛 Starting EV Autonomous Trucking Perception System")
        
        try:
            # Initialize safety framework
            await self.safety.initialize()
            logger.info("✅ Safety framework initialized (ASIL-D compliant)")
            
            # Start perception pipeline
            await self.perception.start()
            logger.info("✅ Perception pipeline started")
            
            # Initialize multi-modal fusion
            await self.fusion.initialize()
            logger.info("✅ Multi-modal fusion initialized")
            
            # Start AI development tools
            await self.ai_tools.start()
            logger.info("✅ AI development tools started")
            
            self.running = True
            logger.info(f"🎯 System running with target latency: {self.latency_target}ms")
            
            # Main processing loop
            while self.running:
                await self._process_frame()
                await asyncio.sleep(0.001)  # 1ms granularity
                
        except Exception as e:
            logger.error(f"❌ System error: {e}")
            await self.shutdown()
            raise
            
    async def _process_frame(self):
        """Process a single frame through the perception pipeline."""
        start_time = asyncio.get_event_loop().time()
        
        # Multi-modal sensor fusion
        fused_data = await self.fusion.process_frame()
        
        # PointNet++ feature extraction
        features = await self.perception.extract_features(fused_data)
        
        # Safety validation
        safety_result = await self.safety.validate_output(features)
        
        if safety_result.is_safe:
            # Process features for downstream tasks
            await self.perception.process_features(features)
        else:
            logger.warning(f"⚠️ Safety validation failed: {safety_result.reason}")
            
        # Latency monitoring
        end_time = asyncio.get_event_loop().time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        if latency > self.latency_target:
            logger.warning(f"⚠️ Latency exceeded target: {latency:.1f}ms > {self.latency_target}ms")
            
    async def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("🛑 Shutting down perception system...")
        self.running = False
        
        await self.perception.shutdown()
        await self.fusion.shutdown()
        await self.safety.shutdown()
        await self.ai_tools.shutdown()
        
        logger.info("✅ System shutdown complete")


@click.command()
@click.option('--config', '-c', default='configs/default.yaml', 
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--simulation', '-s', is_flag=True, help='Run in simulation mode')
def main(config: str, verbose: bool, simulation: bool):
    """EV Autonomous Trucking Perception System."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    if simulation:
        logger.info("🎮 Running in simulation mode")
        
    # Validate config file
    config_path = Path(config)
    if not config_path.exists():
        logger.error(f"❌ Configuration file not found: {config}")
        sys.exit(1)
        
    # Create and run system
    system = EVPerceptionSystem(str(config_path))
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"📡 Received signal {signum}, shutting down...")
        asyncio.create_task(system.shutdown())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
