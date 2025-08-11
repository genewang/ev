#!/usr/bin/env python3
"""
Setup script for EV Autonomous Trucking Perception System.

Provides easy installation and development setup for the perception system.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="ev-perception",
    version="1.0.0",
    author="EV Perception Team",
    author_email="perception@ev.com",
    description="Scalable, cost-effective AI building stack for autonomous driving systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ev/ev-perception",
    project_urls={
        "Bug Reports": "https://github.com/ev/ev-perception/issues",
        "Source": "https://github.com/ev/ev-perception",
        "Documentation": "https://ev-perception.readthedocs.io/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Sensors",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
            "torch-scatter>=2.1.0",
            "torch-sparse>=0.6.0",
            "torch-cluster>=1.6.0",
            "torch-spline-conv>=1.2.0",
        ],
        "ros2": [
            "rclpy>=0.1.0",
            "sensor-msgs>=4.2.0",
            "geometry-msgs>=4.2.0",
            "nav-msgs>=4.2.0",
        ],
        "cloud": [
            "boto3>=1.28.0",
            "kubernetes>=26.1.0",
            "mlflow>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ev-perception=src.main:main",
            "ev-sim=src.simulation.run_sim:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "autonomous driving",
        "perception",
        "lidar",
        "computer vision",
        "point cloud",
        "neural networks",
        "safety",
        "ai",
        "machine learning",
        "robotics",
        "ros2",
    ],
)
