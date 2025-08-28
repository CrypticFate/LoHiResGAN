#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for LoHiResGAN Training with ALL Slices Evaluation
Contains all configuration parameters and paths
"""

import tensorflow as tf
import os

# ==================== TRAINING CONFIGURATION ====================
TRAINING_MODE = True
SEQUENCE_TYPE = 'T1'
EPOCHS = 20  # Reduced epochs due to comprehensive evaluation
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
MAX_SUBJECTS = None  # Use ALL subjects for both training and evaluation

# ==================== PATHS CONFIGURATION ====================
MODEL_PATH = "/kaggle/input/lohiresganv2/keras/default/1/Trained_Model_T1/Trained_Model_T1"
TRAINING_DATA_DIR = "/kaggle/input/d/sakeefhossain/lohiresgan/Training data"
OUTPUT_DIR = "/kaggle/working/Synt_Output"
SAMPLE_OUTPUT_DIR = "/kaggle/working/sample_outputs"
WORKING_DIR = "/kaggle/working"

# ==================== MODEL CONFIGURATION ====================
INPUT_SHAPE = (256, 256, 1)
IMAGE_SIZE = 256
GENERATOR_L1_LAMBDA = 100  # Weight for L1 loss in generator

# ==================== EVALUATION CONFIGURATION ====================
NUM_SAMPLE_SUBJECTS = 5  # Number of subjects to save samples for each epoch
SAVE_MODEL_EVERY_N_EPOCHS = 5  # Save model every N epochs
PRINT_BATCH_EVERY_N = 50  # Print batch progress every N batches

# ==================== GPU CONFIGURATION ====================
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
            return True
        else:
            print("No GPUs found, using CPU")
            return False
    except Exception as e:
        print(f"GPU initialization failed: {e}")
        print("Falling back to CPU")
        tf.config.set_visible_devices([], 'GPU')
        return False

# ==================== DIRECTORY SETUP ====================
def setup_directories():
    """Create necessary output directories"""
    directories = [OUTPUT_DIR, SAMPLE_OUTPUT_DIR, WORKING_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ready: {directory}")

# ==================== VALIDATION ====================
def validate_config():
    """Validate configuration parameters"""
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert EPOCHS > 0, "EPOCHS must be positive"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
    assert SEQUENCE_TYPE in ['T1', 'T2', 'FLAIR'], "SEQUENCE_TYPE must be T1, T2, or FLAIR"
    assert len(INPUT_SHAPE) == 3, "INPUT_SHAPE must be 3D (height, width, channels)"
    
    print("✅ Configuration validation passed")

if __name__ == "__main__":
    print("LoHiResGAN Configuration")
    print("=" * 50)
    print(f"Training Mode: {TRAINING_MODE}")
    print(f"Sequence Type: {SEQUENCE_TYPE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Subjects: {MAX_SUBJECTS if MAX_SUBJECTS else 'ALL'}")
    print("=" * 50)
    
    validate_config()
    setup_directories()
    gpu_available = configure_gpu()
    print(f"GPU Available: {gpu_available}")