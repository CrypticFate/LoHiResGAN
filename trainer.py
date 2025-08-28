#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Trainer class for LoHiResGAN with ALL Slices Evaluation
Orchestrates the training process with comprehensive evaluation
"""

import tensorflow as tf
import numpy as np
from .config import (
    EPOCHS, BATCH_SIZE, SEQUENCE_TYPE, MAX_SUBJECTS, 
    NUM_SAMPLE_SUBJECTS, SAVE_MODEL_EVERY_N_EPOCHS, PRINT_BATCH_EVERY_N
)
from .data_generator import MemoryEfficientDataGenerator
from .model_loader import ModelLoader, LossFunctions, TrainingStep
from .metrics_calculator import ComprehensiveMetricsCalculator
from .utils import ImageUtils, MemoryManager, ProgressTracker, ModelSaver, FileManager

class LoHiResGANTrainer:
    """Main trainer class for LoHiResGAN with comprehensive evaluation"""
    
    def __init__(self, training_data_dir=None, model_path=None, sequence_type=None):
        # Configuration
        self.sequence_type = sequence_type or SEQUENCE_TYPE
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.max_subjects = MAX_SUBJECTS
        
        # Initialize components
        self.data_generator = None
        self.model_loader = ModelLoader(model_path)
        self.metrics_calculator = ComprehensiveMetricsCalculator()
        self.progress_tracker = ProgressTracker()
        self.model_saver = ModelSaver()
        
        # Models and training components
        self.generator = None
        self.discriminator = None
        self.training_step = None
        
        print("="*60)
        print("LOHIRESGAN TRAINER INITIALIZED")
        print(f"Sequence: {self.sequence_type}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Max Subjects: {self.max_subjects if self.max_subjects else 'ALL'}")
        print("="*60)
    
    def setup_data_generator(self, training_data_dir=None):
        """Initialize the data generator"""
        print("Setting up data generator...")
        
        self.data_generator = MemoryEfficientDataGenerator(
            training_data_dir=training_data_dir,
            sequence_type=self.sequence_type,
            max_subjects=self.max_subjects
        )
        
        if len(self.data_generator) == 0:
            raise ValueError("No valid subjects found in data generator!")
        
        print(f"✅ Data generator ready with {len(self.data_generator)} subjects")
        return self.data_generator
    
    def setup_models(self):
        """Load and setup all models"""
        print("Setting up models...")
        
        # Load generator and build discriminator
        self.generator = self.model_loader.load_pretrained_generator()
        self.discriminator = self.model_loader.build_discriminator()
        
        if self.generator is None:
            raise ValueError("Failed to load generator model!")
        
        # Create optimizers and loss functions
        generator_optimizer, discriminator_optimizer = self.model_loader.create_optimizers()
        loss_functions = LossFunctions()
        
        # Create training step handler
        self.training_step = TrainingStep(
            self.generator, self.discriminator,
            generator_optimizer, discriminator_optimizer,
            loss_functions
        )
        
        print("✅ Models and training components ready")
        return self.generator, self.discriminator
    
    def train_epoch(self, epoch):
        """Train for one epoch with comprehensive evaluation"""
        print(f"\nEpoch {epoch+1}/{self.epochs}")
        
        # Reset progress tracking
        self.progress_tracker.reset_batch_count()
        
        # Training metrics
        gen_total_loss_avg = tf.keras.metrics.Mean()
        gen_gan_loss_avg = tf.keras.metrics.Mean()
        gen_l1_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()
        
        # Train on all subjects
        for subject_idx in range(len(self.data_generator)):
            subject_info = self.data_generator.get_subject_info(subject_idx)
            subject_id = subject_info['subject_id']
            
            print(f"  Training on {subject_id}...")
            
            # Load subject data
            low_slices, high_slices, num_slices = self.data_generator.load_subject_data(subject_idx)
            
            if low_slices is not None and num_slices > 0:
                # Training phase - process all slices
                self._train_subject_slices(
                    low_slices, high_slices, num_slices,
                    gen_total_loss_avg, gen_gan_loss_avg, gen_l1_loss_avg, disc_loss_avg
                )
                
                # Evaluation phase - evaluate middle slice
                self._evaluate_middle_slice(subject_id, low_slices, high_slices, num_slices)
            
            # Cleanup memory after each subject
            MemoryManager.cleanup_variables(low_slices, high_slices)
        
        # Print epoch summary
        self.progress_tracker.print_epoch_summary(
            epoch + 1,
            gen_total_loss_avg.result(),
            gen_gan_loss_avg.result(),
            gen_l1_loss_avg.result(),
            disc_loss_avg.result()
        )
        
        return {
            'gen_total_loss': float(gen_total_loss_avg.result()),
            'gen_gan_loss': float(gen_gan_loss_avg.result()),
            'gen_l1_loss': float(gen_l1_loss_avg.result()),
            'disc_loss': float(disc_loss_avg.result())
        }
    
    def _train_subject_slices(self, low_slices, high_slices, num_slices,
                             gen_total_avg, gen_gan_avg, gen_l1_avg, disc_avg):
        """Train on all slices of a subject"""
        for i in range(0, num_slices, self.batch_size):
            end_idx = min(i + self.batch_size, num_slices)
            
            batch_input = low_slices[i:end_idx]
            batch_target = high_slices[i:end_idx]
            
            # Execute training step
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = \
                self.training_step.train_step(batch_input, batch_target)
            
            # Update metrics
            gen_total_avg.update_state(gen_total_loss)
            gen_gan_avg.update_state(gen_gan_loss)
            gen_l1_avg.update_state(gen_l1_loss)
            disc_avg.update_state(disc_loss)
            
            # Update progress tracking
            self.progress_tracker.update_batch_metrics(
                gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss
            )
            
            # Print progress
            self.progress_tracker.print_batch_progress(
                self.progress_tracker.batch_count, gen_total_loss, disc_loss, PRINT_BATCH_EVERY_N
            )
    
    def _evaluate_middle_slice(self, subject_id, low_slices, high_slices, num_slices):
        """Evaluate the middle slice of a subject"""
        print(f"    Evaluating middle slice for {subject_id}...")
        
        # Use middle slice for evaluation
        middle_idx = num_slices // 2
        slice_input = low_slices[middle_idx:middle_idx+1]
        slice_target = high_slices[middle_idx:middle_idx+1]
        
        # Generate output for this slice
        slice_generated = self.generator(slice_input, training=False)
        
        # Convert to numpy if needed
        target_array = slice_target.numpy() if hasattr(slice_target, 'numpy') else slice_target
        generated_array = slice_generated.numpy() if hasattr(slice_generated, 'numpy') else slice_generated
        
        # Calculate metrics for this slice
        slice_filename = f"{subject_id}_{self.sequence_type}.nii.gz"
        slice_metrics = self.metrics_calculator.add_metrics(
            target_array, generated_array, slice_filename
        )
        
        self.progress_tracker.increment_slices_evaluated()
        
        print(f"    SUCCESS {subject_id} - MIDDLE SLICE:")
        print(f"      MAE: {slice_metrics['mae']:.6f}, SSIM: {slice_metrics['ssim']:.6f}, PSNR: {slice_metrics['psnr']:.6f}")
        
        return slice_metrics
    
    def save_sample_images(self, epoch):
        """Save sample images for multiple subjects"""
        print(f"  Saving samples from {NUM_SAMPLE_SUBJECTS} subjects (middle slice each)...")
        
        total_samples_saved = 0
        
        # Process sample subjects
        for subject_idx in range(min(NUM_SAMPLE_SUBJECTS, len(self.data_generator))):
            subject_info = self.data_generator.get_subject_info(subject_idx)
            subject_id = subject_info['subject_id']
            
            # Get middle slice for this subject
            slice_input, slice_target, _ = self.data_generator.get_middle_slice(subject_idx)
            
            if slice_input is not None:
                try:
                    # Generate output
                    slice_generated = self.generator(slice_input, training=False)
                    
                    # Save images
                    success = ImageUtils.save_sample_images(
                        epoch + 1, subject_id, slice_input, slice_target, slice_generated
                    )
                    
                    if success:
                        total_samples_saved += 1
                        
                except Exception as e:
                    print(f"    Error saving {subject_id}: {e}")
            
            # Cleanup
            MemoryManager.cleanup_variables(slice_input, slice_target)
        
        print(f"  SUCCESS: {total_samples_saved} sample sets saved")
        return total_samples_saved
    
    def save_model_checkpoint(self, epoch):
        """Save model checkpoint if needed"""
        if (epoch + 1) % SAVE_MODEL_EVERY_N_EPOCHS == 0:
            self.model_saver.save_model(self.generator, self.sequence_type, epoch + 1)
    
    def save_final_model(self):
        """Save the final trained model"""
        return self.model_saver.save_model(self.generator, self.sequence_type, is_final=True)
    
    def save_final_metrics(self):
        """Save comprehensive metrics to CSV"""
        csv_filename = FileManager.get_csv_filename(self.sequence_type)
        final_df = self.metrics_calculator.save_metrics_to_csv(csv_filename)
        
        print(f"\n💾 Comprehensive metrics saved to: {csv_filename}")
        print(f"📊 Total slices evaluated: {self.metrics_calculator.get_metrics_count()}")
        
        return csv_filename, final_df
    
    def train(self, training_data_dir=None):
        """Main training loop"""
        print("🚀 Starting LoHiResGAN Training with ALL Slices Evaluation...")
        
        # Setup components
        self.setup_data_generator(training_data_dir)
        self.setup_models()
        
        # Training loop
        for epoch in range(self.epochs):
            # Train one epoch
            epoch_metrics = self.train_epoch(epoch)
            
            # Save sample images
            self.save_sample_images(epoch)
            
            # Save model checkpoint
            self.save_model_checkpoint(epoch)
            
            # Force memory cleanup
            MemoryManager.cleanup()
        
        # Save final model and metrics
        final_model_path = self.save_final_model()
        csv_filename, final_df = self.save_final_metrics()
        
        # Print completion summary
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED WITH ALL SLICES EVALUATION!")
        print(f"📁 Final model: {final_model_path}")
        print(f"📊 Metrics CSV: {csv_filename}")
        print(f"🔢 Total slices evaluated: {self.metrics_calculator.get_metrics_count()}")
        print("="*60)
        
        return {
            'model_path': final_model_path,
            'csv_filename': csv_filename,
            'final_metrics': final_df,
            'total_slices': self.metrics_calculator.get_metrics_count()
        }

if __name__ == "__main__":
    # Test the trainer
    print("Testing LoHiResGANTrainer...")
    
    try:
        # Create trainer instance
        trainer = LoHiResGANTrainer(sequence_type='T1')
        print(f"✅ Trainer created successfully for sequence: {trainer.sequence_type}")
        
        # Test component initialization (will fail without actual data/model)
        print("Note: Full training test requires actual data and model files")
        
    except Exception as e:
        print(f"Test failed (expected without data/model): {e}")
    
    print("✅ Trainer structure test completed!")