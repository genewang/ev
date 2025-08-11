#!/usr/bin/env python3
"""
Quick start script for EV Autonomous Trucking Perception System.

This script helps users quickly get started with the perception system
by setting up the environment and running basic tests.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path
from loguru import logger


def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 9):
        logger.error("❌ Python 3.9+ required. Current version: {}.{}".format(
            sys.version_info.major, sys.version_info.minor))
        return False
    logger.info("✅ Python version check passed")
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        logger.warning("⚠️ PyTorch not available")
        
    try:
        import numpy
        logger.info(f"✅ NumPy {numpy.__version__} available")
    except ImportError:
        logger.warning("⚠️ NumPy not available")
        
    try:
        import opencv
        logger.info(f"✅ OpenCV {opencv.__version__} available")
    except ImportError:
        logger.warning("⚠️ OpenCV not available")
        
    return True


def install_dependencies():
    """Install required dependencies."""
    logger.info("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False


def setup_environment():
    """Set up the development environment."""
    logger.info("🔧 Setting up development environment...")
    
    # Create necessary directories
    directories = ["models", "logs", "data", "configs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
        
    # Copy default config if it doesn't exist
    if not Path("configs/default.yaml").exists():
        logger.warning("⚠️ Default configuration not found. Please ensure configs/default.yaml exists.")
        
    logger.info("✅ Environment setup complete")
    return True


def run_tests():
    """Run the test suite."""
    logger.info("🧪 Running tests...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Tests passed successfully")
            return True
        else:
            logger.warning("⚠️ Some tests failed")
            logger.info("Test output:")
            print(result.stdout)
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run tests: {e}")
        return False


async def run_simulation():
    """Run a quick simulation."""
    logger.info("🎮 Running quick simulation...")
    
    try:
        # Import and run simulation
        sys.path.append("src")
        from simulation.run_sim import SimulationEnvironment
        
        sim_env = SimulationEnvironment("configs/default.yaml")
        await sim_env.start_simulation(duration=10.0, fps=10)  # 10s, 10 FPS
        
        logger.info("✅ Simulation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        return False


def show_next_steps():
    """Show next steps for users."""
    logger.info("🚀 Next Steps:")
    logger.info("1. Review the configuration in configs/default.yaml")
    logger.info("2. Run the main system: python src/main.py")
    logger.info("3. Run simulation: python src/simulation/run_sim.py")
    logger.info("4. Check the README.md for detailed documentation")
    logger.info("5. Run tests: python -m pytest tests/")
    
    logger.info("\n📚 Useful Commands:")
    logger.info("  # Install in development mode")
    logger.info("  pip install -e .[dev]")
    logger.info("")
    logger.info("  # Run with Docker")
    logger.info("  docker-compose -f docker/docker-compose.yml up")
    logger.info("")
    logger.info("  # Run specific tests")
    logger.info("  python -m pytest tests/test_perception.py -v")


def main():
    """Main quick start function."""
    logger.info("🚛 EV Autonomous Trucking Perception System - Quick Start")
    logger.info("=" * 70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
        
    # Check dependencies
    check_dependencies()
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
        
    # Install dependencies if needed
    if not Path("requirements.txt").exists():
        logger.error("❌ requirements.txt not found")
        sys.exit(1)
        
    # Ask user if they want to install dependencies
    try:
        response = input("\n📦 Install dependencies? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            if not install_dependencies():
                sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Setup interrupted")
        sys.exit(0)
        
    # Run tests if dependencies are available
    try:
        import torch
        import numpy
        run_tests()
    except ImportError:
        logger.warning("⚠️ Skipping tests - dependencies not available")
        
    # Show next steps
    show_next_steps()
    
    logger.info("\n🎉 Quick start complete! The EV perception system is ready to use.")
    logger.info("For more information, see the README.md file.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Quick start failed: {e}")
        sys.exit(1)
