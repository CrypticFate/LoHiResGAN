#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for LoHiResGAN
Contains helper functions for image processing, saving, and visualization
"""

import numpy as np
import nibabel as nib
import os
import gc
from PIL import Image
from .config import SAMPLE_OUTPUT_DIR, WORKING_DIR

class ImageUtils:
    """Utility functions for image processing and saving"""
    
    @staticmethod
    def to_png_data(data):
        """Convert tensor data to PNG-compatible format"""
        data = data.squeeze()  # Remove extra dimensions
        
        # Normalize to [0, 1] range
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        
        # Convert to 0-255 range for PNG
        return (data * 255).astype(np.uint8)
    
    @staticmethod
    def save_sample_images(epoch, subject_id, slice_input, slice_target, slice_generated, 
                          sample_dir=None, save_nifti=True, save_png=True):
        """Save sample images in both PNG and NIfTI formats"""
        if sample_dir is None:
            sample_dir = SAMPLE_OUTPUT_DIR
        
        # Create epoch-specific directory
        epoch_dir = os.path.join(sample_dir, f"epoch{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Create base filename
        base_filename = f"{subject_id}_middle"
        
        try:
            # Save as PNG images
            if save_png:
                Image.fromarray(ImageUtils.to_png_data(slice_input)).save(
                    f"{epoch_dir}/{base_filename}_input.png")
                Image.fromarray(ImageUtils.to_png_data(slice_target)).save(
                    f"{epoch_dir}/{base_filename}_target.png")
                Image.fromarray(ImageUtils.to_png_data(slice_generated)).save(
                    f"{epoch_dir}/{base_filename}_generated.png")
            
            # Save as NIfTI files for medical viewing
            if save_nifti:
                input_nii = nib.Nifti1Image(slice_input.squeeze(), affine=np.eye(4))
                target_nii = nib.Nifti1Image(slice_target.squeeze(), affine=np.eye(4))
                generated_nii = nib.Nifti1Image(slice_generated.squeeze(), affine=np.eye(4))
                
                nib.save(input_nii, f"{epoch_dir}/{base_filename}_input.nii.gz")
                nib.save(target_nii, f"{epoch_dir}/{base_filename}_target.nii.gz")
                nib.save(generated_nii, f"{epoch_dir}/{base_filename}_generated.nii.gz")
            
            return True
            
        except Exception as e:
            print(f"Error saving images for {subject_id}: {e}")
            return False

class MemoryManager:
    """Utility functions for memory management"""
    
    @staticmethod
    def cleanup():
        """Force garbage collection"""
        gc.collect()
    
    @staticmethod
    def cleanup_variables(*variables):
        """Delete variables and force garbage collection"""
        for var in variables:
            if var is not None:
                del var
        gc.collect()

class ProgressTracker:
    """Tracks and displays training progress"""
    
    def __init__(self):
        self.epoch_metrics = []
        self.batch_count = 0
        self.total_slices_evaluated = 0
    
    def update_batch_metrics(self, gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss):
        """Update metrics for current batch"""
        self.batch_count += 1
        
        # Store current batch metrics (could be expanded for detailed tracking)
        current_metrics = {
            'batch': self.batch_count,
            'gen_total_loss': float(gen_total_loss),
            'gen_gan_loss': float(gen_gan_loss),
            'gen_l1_loss': float(gen_l1_loss),
            'disc_loss': float(disc_loss)
        }
        
        return current_metrics
    
    def print_batch_progress(self, batch_num, gen_loss, disc_loss, print_every=50):
        """Print batch progress at specified intervals"""
        if batch_num % print_every == 0:
            print(f"    Batch {batch_num}: Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
    
    def print_epoch_summary(self, epoch, gen_total_avg, gen_gan_avg, gen_l1_avg, disc_avg):
        """Print comprehensive epoch summary"""
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Gen Total: {gen_total_avg:.4f}")
        print(f"  Train Gen GAN: {gen_gan_avg:.4f}")
        print(f"  Train Gen L1: {gen_l1_avg:.4f}")
        print(f"  Train Disc: {disc_avg:.4f}")
        print(f"  Total batches: {self.batch_count}")
        print(f"  Total slices evaluated: {self.total_slices_evaluated}")
    
    def increment_slices_evaluated(self, count=1):
        """Increment the count of evaluated slices"""
        self.total_slices_evaluated += count
    
    def reset_batch_count(self):
        """Reset batch count for new epoch"""
        self.batch_count = 0

class ModelSaver:
    """Handles model saving operations"""
    
    def __init__(self, working_dir=None):
        self.working_dir = working_dir or WORKING_DIR
    
    def save_model(self, model, sequence_type, epoch=None, is_final=False):
        """Save model with appropriate naming"""
        try:
            if is_final:
                save_path = f"{self.working_dir}/MemoryEfficient_LoHiResGAN_{sequence_type}_Final"
                print(f"Saving final model: {save_path}")
            else:
                save_path = f"{self.working_dir}/MemoryEfficient_LoHiResGAN_{sequence_type}_Epoch_{epoch}"
                print(f"Saving model for epoch {epoch}: {save_path}")
            
            model.save(save_path)
            print(f"✅ Model saved successfully: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Could not save model: {e}")
            return None

class FileManager:
    """Handles file operations and directory management"""
    
    @staticmethod
    def ensure_directory_exists(directory):
        """Create directory if it doesn't exist"""
        os.makedirs(directory, exist_ok=True)
        return directory
    
    @staticmethod
    def get_csv_filename(sequence_type, working_dir=None):
        """Generate CSV filename for metrics"""
        if working_dir is None:
            working_dir = WORKING_DIR
        
        return f"{working_dir}/LoHiResGAN_{sequence_type}_ALL_SLICES_metrics.csv"
    
    @staticmethod
    def validate_file_exists(filepath):
        """Check if file exists"""
        return os.path.exists(filepath)

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test ImageUtils
    test_data = np.random.rand(256, 256)
    png_data = ImageUtils.to_png_data(test_data)
    print(f"PNG conversion test: {png_data.shape}, dtype: {png_data.dtype}")
    
    # Test ProgressTracker
    tracker = ProgressTracker()
    metrics = tracker.update_batch_metrics(1.5, 0.8, 0.7, 1.2)
    print(f"Progress tracking test: {metrics}")
    
    # Test MemoryManager
    test_var = np.random.rand(100, 100)
    MemoryManager.cleanup_variables(test_var)
    print("Memory cleanup test completed")
    
    # Test FileManager
    test_filename = FileManager.get_csv_filename("T1")
    print(f"CSV filename generation test: {test_filename}")
    
    print("✅ Utility functions test completed successfully!")