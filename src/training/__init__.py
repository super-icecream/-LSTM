"""Training module exports for DLFE-LSTM-WSI."""

from .trainer import GPUOptimizedTrainer
from .validator import Validator, MultiModelValidator
from .adaptive_optimizer import AdaptiveOptimizer
from .diagnostics import TrainingDiagnostics

__all__ = [
    "GPUOptimizedTrainer",
    "Validator",
    "MultiModelValidator",
    "AdaptiveOptimizer",
    "TrainingDiagnostics",
]

__version__ = "1.0.0"
__author__ = "DLFE-LSTM-WSI Team"
__description__ = "GPU-aware trainers and validators for DLFE-LSTM-WSI"

