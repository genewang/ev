"""
Deployment targets. Perception and language-model runtimes are different stacks.
"""

from enum import Enum


class DeploymentTarget(Enum):
    """Where a model is allowed to run."""

    PYTORCH = "pytorch"
    ONNX_RUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"
    COREML_ANE = "coreml_ane"
    TENSORRT_LLM = "tensorrt_llm"


PERCEPTION_TARGETS = (
    DeploymentTarget.PYTORCH,
    DeploymentTarget.ONNX_RUNTIME,
    DeploymentTarget.TENSORRT,
    DeploymentTarget.COREML_ANE,
)

LANGUAGE_TARGETS = (DeploymentTarget.TENSORRT_LLM,)


def allows_perception(target: DeploymentTarget) -> bool:
    """TensorRT-LLM is for token generation, not the sensor-fusion loop."""
    return target in PERCEPTION_TARGETS


def target_notes(target: DeploymentTarget) -> str:
    notes = {
        DeploymentTarget.PYTORCH: "Reference implementation used by the in-process loop.",
        DeploymentTarget.ONNX_RUNTIME: "Portable export path. Measure accuracy after export.",
        DeploymentTarget.TENSORRT: "NVIDIA GPU engine for perception. Build it on the target device.",
        DeploymentTarget.COREML_ANE: "Apple Silicon path. Separate from TensorRT; do not mix the claims.",
        DeploymentTarget.TENSORRT_LLM: "Language-model serving only. Keep it off the real-time perception path.",
    }
    return notes[target]
