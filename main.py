#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main execution file for Modular LoHiResGAN Training with ALL Slices Evaluation
This is the entry point that orchestrates all components
"""

import sys
import os
import argparse
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TRAINING_MODE, SEQUENCE_TYPE, EPOCHS, MAX_SUBJECTS
from trainer import LoHiResGANTrainer
from utils import FileManager
import modular_lohiresgan as mlg

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LoHiResGAN Training with ALL Slices Evaluation')
    
    parser.add_argument('--training_mode', type=bool, default=TRAINING_MODE,
                       help='Enable training mode (default: from config)')
    
    parser.add_argument('--sequence_type', type=str, default=SEQUENCE_TYPE,
                       choices=['T1', 'T2', 'FLAIR'],
                       help='MRI sequence type to process')
    
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    
    parser.add_argument('--max_subjects', type=int, default=MAX_SUBJECTS,
                       help='Maximum number of subjects to process (None for all)')
    
    parser.add_argument('--training_data_dir', type=str, default=None,
                       help='Path to training data directory')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model')
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              LoHiResGAN Modular Training System              ║
    ║                                                              ║
    ║        Low-to-High Resolution MRI Enhancement using          ║
    ║              Generative Adversarial Networks                 ║
    ║                                                              ║
    ║                  With ALL Slices Evaluation                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_configuration(args):
    """Print current configuration"""
    print("🔧 CONFIGURATION")
    print("=" * 60)
    print(f"Training Mode:     {args.training_mode}")
    print(f"Sequence Type:     {args.sequence_type}")
    print(f"Epochs:           {args.epochs}")
    print(f"Max Subjects:     {args.max_subjects if args.max_subjects else 'ALL'}")
    print(f"Training Data:    {args.training_data_dir or 'From config'}")
    print(f"Model Path:       {args.model_path or 'From config'}")
    print(f"Output Dir:       {args.output_dir or 'From config'}")
    print(f"Verbose:          {args.verbose}")
    print("=" * 60)

def validate_inputs(args):
    """Validate input arguments"""
    print("🔍 VALIDATING INPUTS...")
    
    # Check training data directory if provided
    if args.training_data_dir and not os.path.exists(args.training_data_dir):
        raise ValueError(f"Training data directory not found: {args.training_data_dir}")
    
    # Check model path if provided
    if args.model_path and not os.path.exists(args.model_path):
        raise ValueError(f"Model path not found: {args.model_path}")
    
    # Validate sequence type
    if args.sequence_type not in ['T1', 'T2', 'FLAIR']:
        raise ValueError(f"Invalid sequence type: {args.sequence_type}")
    
    # Validate epochs
    if args.epochs <= 0:
        raise ValueError(f"Epochs must be positive: {args.epochs}")
    
    print("✅ Input validation passed")

def run_training(args):
    """Execute the training process"""
    print("🚀 STARTING TRAINING PROCESS...")
    
    start_time = datetime.now()
    
    try:
        # Create trainer instance
        trainer = LoHiResGANTrainer(
            training_data_dir=args.training_data_dir,
            model_path=args.model_path,
            sequence_type=args.sequence_type
        )
        
        # Override configuration if provided via arguments
        if args.epochs != EPOCHS:
            trainer.epochs = args.epochs
        if args.max_subjects is not None:
            trainer.max_subjects = args.max_subjects
        
        # Execute training
        results = trainer.train(training_data_dir=args.training_data_dir)
        
        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Print results summary
        print_results_summary(results, training_duration, args)
        
        return results
        
    except Exception as e:
        print(f"❌ TRAINING FAILED: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

def run_evaluation_only(args):
    """Run evaluation only (no training)"""
    print("📊 STARTING EVALUATION PROCESS...")
    
    start_time = datetime.now()
    
    try:
        # Create trainer instance
        trainer = LoHiResGANTrainer(
            training_data_dir=args.training_data_dir,
            model_path=args.model_path,
            sequence_type=args.sequence_type
        )
        
        # Setup components
        trainer.setup_data_generator(args.training_data_dir)
        trainer.setup_models()
        
        # Run evaluation on all subjects
        print("Evaluating all subjects...")
        for subject_idx in range(len(trainer.data_generator)):
            subject_info = trainer.data_generator.get_subject_info(subject_idx)
            subject_id = subject_info['subject_id']
            
            # Get middle slice and evaluate
            slice_input, slice_target, _ = trainer.data_generator.get_middle_slice(subject_idx)
            
            if slice_input is not None:
                # Generate output
                slice_generated = trainer.generator(slice_input, training=False)
                
                # Calculate metrics
                target_array = slice_target.numpy() if hasattr(slice_target, 'numpy') else slice_target
                generated_array = slice_generated.numpy() if hasattr(slice_generated, 'numpy') else slice_generated
                
                slice_filename = f"{subject_id}_{args.sequence_type}.nii.gz"
                metrics = trainer.metrics_calculator.add_metrics(
                    target_array, generated_array, slice_filename
                )
                
                print(f"✅ {subject_id}: MAE={metrics['mae']:.6f}, SSIM={metrics['ssim']:.6f}, PSNR={metrics['psnr']:.6f}")
        
        # Save metrics
        csv_filename, final_df = trainer.save_final_metrics()
        
        end_time = datetime.now()
        evaluation_duration = end_time - start_time
        
        results = {
            'csv_filename': csv_filename,
            'final_metrics': final_df,
            'total_slices': trainer.metrics_calculator.get_metrics_count(),
            'evaluation_only': True
        }
        
        print_results_summary(results, evaluation_duration, args)
        
        return results
        
    except Exception as e:
        print(f"❌ EVALUATION FAILED: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

def print_results_summary(results, duration, args):
    """Print comprehensive results summary"""
    print("\n" + "🎉 PROCESS COMPLETED SUCCESSFULLY!" + "\n")
    print("📋 RESULTS SUMMARY")
    print("=" * 60)
    
    if results.get('evaluation_only', False):
        print("Mode:              Evaluation Only")
    else:
        print("Mode:              Training + Evaluation")
        if 'model_path' in results:
            print(f"Final Model:       {results['model_path']}")
    
    print(f"Sequence Type:     {args.sequence_type}")
    print(f"Total Duration:    {duration}")
    print(f"Slices Evaluated:  {results['total_slices']}")
    
    if 'csv_filename' in results:
        print(f"Metrics CSV:       {results['csv_filename']}")
    
    # Print top metrics if available
    if 'final_metrics' in results and results['final_metrics'] is not None:
        df = results['final_metrics']
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"Average MAE:       {df['mae'].mean():.6f}")
        print(f"Average PSNR:      {df['psnr'].mean():.2f} dB")
        print(f"Average SSIM:      {df['ssim'].mean():.4f}")
        print(f"Average Correlation: {df['corr'].mean():.4f}")
        
        # Show best performing subject
        best_subject = df.loc[df['ssim'].idxmax()]
        print(f"\n🏆 BEST PERFORMING SUBJECT:")
        print(f"Subject:           {best_subject['filename']}")
        print(f"SSIM:              {best_subject['ssim']:.4f}")
        print(f"PSNR:              {best_subject['psnr']:.2f} dB")
        print(f"MAE:               {best_subject['mae']:.6f}")
    
    print("=" * 60)

def main():
    """Main execution function"""
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Print configuration
    print_configuration(args)
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Quick environment setup
        mlg.quick_setup()
        
        # Execute based on mode
        if args.training_mode:
            results = run_training(args)
        else:
            results = run_evaluation_only(args)
        
        # Exit with appropriate code
        if results is not None:
            print("\n✅ Process completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Process failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()