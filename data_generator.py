#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-Efficient Data Generator for LoHiResGAN
Handles loading and preprocessing of POCEMR subjects
"""

import nibabel as nib
import numpy as np
import cv2
import os
import glob
import gc
from .config import TRAINING_DATA_DIR, IMAGE_SIZE

class MemoryEfficientDataGenerator:
    """Memory-efficient data generator for all subjects"""
    
    def __init__(self, training_data_dir=None, sequence_type='T1', max_subjects=None):
        self.training_data_dir = training_data_dir or TRAINING_DATA_DIR
        self.sequence_type = sequence_type
        self.subject_files = []
        
        # Find all subject folders
        subject_folders = glob.glob(os.path.join(self.training_data_dir, 'POCEMR*'))
        subject_folders.sort()  # Ensure consistent ordering
        print(f"Found {len(subject_folders)} POCEMR* subjects")
        
        # Use ALL subjects for both training and evaluation
        if max_subjects:
            subject_folders = subject_folders[:max_subjects]
            print(f"Using {len(subject_folders)} subjects")
        else:
            print(f"Using ALL {len(subject_folders)} subjects")
        
        # Collect valid file pairs
        self._collect_valid_subjects(subject_folders)
        
        if len(self.subject_files) > 0:
            print(f"Will process subjects from {self.subject_files[0][2]} to {self.subject_files[-1][2]}")
    
    def _collect_valid_subjects(self, subject_folders):
        """Collect valid subject file pairs"""
        for subject_folder in subject_folders:
            subject_id = os.path.basename(subject_folder)
            
            low_field_path = os.path.join(subject_folder, '64mT', f'{subject_id}_{self.sequence_type}.nii')
            high_field_path = os.path.join(subject_folder, '3T', f'{subject_id}_{self.sequence_type}.nii')
            
            if os.path.exists(low_field_path) and os.path.exists(high_field_path):
                self.subject_files.append((low_field_path, high_field_path, subject_id))
                print(f"Added: {subject_id}")
        
        print(f"Total valid subjects: {len(self.subject_files)}")
    
    def load_subject_data(self, subject_idx):
        """Load data for a single subject"""
        if subject_idx >= len(self.subject_files):
            raise IndexError(f"Subject index {subject_idx} out of range")
        
        low_field_path, high_field_path, subject_id = self.subject_files[subject_idx]
        
        try:
            # Load volumes
            low_field_img = nib.load(low_field_path)
            high_field_img = nib.load(high_field_path)
            
            low_field_data = low_field_img.get_fdata()
            high_field_data = high_field_img.get_fdata()
            
            # Normalize to [-1, 1] range
            low_field_normalized = self._normalize_to_range(low_field_data, -1, 1)
            high_field_normalized = self._normalize_to_range(high_field_data, -1, 1)
            
            # Process slices
            low_slices, high_slices = self._process_slices(low_field_normalized, high_field_normalized)
            
            # Convert to arrays and add channel dimension
            low_slices = np.array(low_slices, dtype=np.float32)
            high_slices = np.array(high_slices, dtype=np.float32)
            
            low_slices = np.expand_dims(low_slices, axis=-1)
            high_slices = np.expand_dims(high_slices, axis=-1)
            
            return low_slices, high_slices, len(low_slices)
            
        except Exception as e:
            print(f"Error loading {subject_id}: {e}")
            return None, None, 0
    
    def _normalize_to_range(self, data, min_val, max_val):
        """Normalize data to specified range"""
        data_min, data_max = data.min(), data.max()
        if data_max == data_min:
            return np.zeros_like(data)
        
        normalized = (data - data_min) / (data_max - data_min)
        return normalized * (max_val - min_val) + min_val
    
    def _process_slices(self, low_field_data, high_field_data):
        """Process all slices from the volume"""
        low_slices = []
        high_slices = []
        
        for i in range(low_field_data.shape[2]):
            low_slice = low_field_data[:, :, i]
            high_slice = high_field_data[:, :, i]
            
            # Apply transformations
            low_slice = cv2.rotate(low_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)
            high_slice = cv2.rotate(high_slice, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            low_slice = cv2.resize(low_slice, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            high_slice = cv2.resize(high_slice, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            
            # Skip empty slices (low variation indicates empty slice)
            if np.std(low_slice) > 0.1 and np.std(high_slice) > 0.1:
                low_slices.append(low_slice)
                high_slices.append(high_slice)
        
        return low_slices, high_slices
    
    def get_subject_info(self, subject_idx):
        """Get information about a specific subject"""
        if subject_idx >= len(self.subject_files):
            return None
        
        low_field_path, high_field_path, subject_id = self.subject_files[subject_idx]
        return {
            'subject_id': subject_id,
            'low_field_path': low_field_path,
            'high_field_path': high_field_path,
            'index': subject_idx
        }
    
    def get_middle_slice(self, subject_idx):
        """Get only the middle slice of a subject (for evaluation)"""
        low_slices, high_slices, num_slices = self.load_subject_data(subject_idx)
        
        if low_slices is not None and num_slices > 0:
            middle_idx = num_slices // 2
            return (
                low_slices[middle_idx:middle_idx+1],
                high_slices[middle_idx:middle_idx+1],
                1
            )
        
        return None, None, 0
    
    def __len__(self):
        """Return the number of subjects"""
        return len(self.subject_files)
    
    def __getitem__(self, idx):
        """Get subject data by index"""
        return self.load_subject_data(idx)
    
    def cleanup_memory(self):
        """Force garbage collection"""
        gc.collect()

if __name__ == "__main__":
    # Test the data generator
    print("Testing MemoryEfficientDataGenerator...")
    
    # Create a test generator (will fail if no data directory exists)
    try:
        generator = MemoryEfficientDataGenerator(sequence_type='T1', max_subjects=2)
        
        if len(generator) > 0:
            print(f"Generator created successfully with {len(generator)} subjects")
            
            # Test loading first subject
            low, high, num_slices = generator.load_subject_data(0)
            if low is not None:
                print(f"First subject loaded: {num_slices} slices")
                print(f"Low field shape: {low.shape}")
                print(f"High field shape: {high.shape}")
            
            # Test middle slice extraction
            low_mid, high_mid, _ = generator.get_middle_slice(0)
            if low_mid is not None:
                print(f"Middle slice extracted: {low_mid.shape}")
        else:
            print("No valid subjects found")
            
    except Exception as e:
        print(f"Test failed (expected if no data directory): {e}")
    
    print("✅ Data generator test completed!")