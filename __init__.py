#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular LoHiResGAN Package
A modular implementation of LoHiResGAN training with ALL slices evaluation
"""

from .config import (
    TRAINING_MODE, SEQUENCE_TYPE, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MODEL_PATH, TRAINING_DATA_DIR, OUTPUT_DIR,
    configure_gpu, setup_directories, validate_config
)

from .data_generator import MemoryEfficientDataGenerator

from .metrics_calculator import ComprehensiveMetricsCalculator

from .model_loader import ModelLoader, LossFunctions, TrainingStep

from .utils import (
    ImageUtils, MemoryManager, ProgressTracker, 
    ModelSaver, FileManager
)

from .trainer import LoHiResGANTrainer

__version__ = "1.0.0"
__author__ = "LoHiResGAN Team"
__description__ = "Modular implementation of LoHiResGAN with comprehensive evaluation"

# Package-level convenience functions
def create_trainer(training_data_dir=None, model_path=None, sequence_type=None):
    """Create a configured trainer instance"""
    return LoHiResGANTrainer(
        training_data_dir=training_data_dir,
        model_path=model_path,
        sequence_type=sequence_type
    )

def quick_setup():
    """Quick setup for training environment"""
    print("🔧 Setting up LoHiResGAN environment...")
    
    # Validate configuration
    validate_config()
    
    # Setup directories
    setup_directories()
    
    # Configure GPU
    gpu_available = configure_gpu()
    
    print(f"✅ Environment setup complete (GPU: {gpu_available})")
    return gpu_available

# Export main classes and functions
__all__ = [
    # Main trainer
    'LoHiResGANTrainer',
    'create_trainer',
    'quick_setup',
    
    # Core components
    'MemoryEfficientDataGenerator',
    'ComprehensiveMetricsCalculator',
    'ModelLoader',
    'LossFunctions',
    'TrainingStep',
    
    # Utilities
    'ImageUtils',
    'MemoryManager',
    'ProgressTracker',
    'ModelSaver',
    'FileManager',
    
    # Configuration
    'TRAINING_MODE',
    'SEQUENCE_TYPE',
    'EPOCHS',
    'BATCH_SIZE',
    'LEARNING_RATE',
    'MODEL_PATH',
    'TRAINING_DATA_DIR',
    'OUTPUT_DIR',
    'configure_gpu',
    'setup_directories',
    'validate_config'
]

if __name__ == "__main__":
    print(f"LoHiResGAN Modular Package v{__version__}")
    print(f"Description: {__description__}")
    print(f"Author: {__author__}")
    print("\nAvailable components:")
    for component in __all__:
        print(f"  - {component}")