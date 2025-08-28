#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Metrics Calculator for LoHiResGAN
Calculates all 8 metrics exactly as in Fateen_lohiresGAN_T1 - metrics.csv.csv
"""

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr

class ComprehensiveMetricsCalculator:
    """
    Calculates all metrics exactly as in Fateen_lohiresGAN_T1 - metrics.csv.csv
    """
    
    def __init__(self):
        self.metrics_data = []
    
    def calculate_mae(self, target, generated):
        """Mean Absolute Error"""
        return np.mean(np.abs(target - generated))
    
    def calculate_mse(self, target, generated):
        """Mean Squared Error"""
        return np.mean((target - generated) ** 2)
    
    def calculate_rmse(self, target, generated):
        """Root Mean Squared Error"""
        mse = self.calculate_mse(target, generated)
        return np.sqrt(mse)
    
    def calculate_nrmse(self, target, generated):
        """Normalized Root Mean Squared Error"""
        rmse = self.calculate_rmse(target, generated)
        target_range = np.max(target) - np.min(target)
        if target_range == 0:
            return 0.0
        return rmse / target_range
    
    def calculate_nmse(self, target, generated):
        """Normalized Mean Squared Error"""
        mse = self.calculate_mse(target, generated)
        target_var = np.var(target)
        if target_var == 0:
            return 0.0
        return mse / target_var
    
    def calculate_psnr(self, target, generated):
        """Peak Signal-to-Noise Ratio"""
        # Normalize to [0, 1] range
        if target.min() < 0:
            target_norm = (target + 1) / 2
            generated_norm = (generated + 1) / 2
        else:
            target_norm = target
            generated_norm = generated
        
        try:
            return psnr(target_norm, generated_norm, data_range=1.0)
        except:
            return 0.0
    
    def calculate_ssim(self, target, generated):
        """Structural Similarity Index"""
        # Normalize to [0, 1] range
        if target.min() < 0:
            target_norm = (target + 1) / 2
            generated_norm = (generated + 1) / 2
        else:
            target_norm = target
            generated_norm = generated
        
        try:
            return ssim(target_norm, generated_norm, data_range=1.0)
        except:
            return 0.0
    
    def calculate_correlation(self, target, generated):
        """Pearson Correlation Coefficient"""
        try:
            corr, _ = pearsonr(target.flatten(), generated.flatten())
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def calculate_all_metrics(self, target, generated, filename):
        """Calculate all metrics for a single image pair"""
        # Ensure images are 2D
        target = target.squeeze()
        generated = generated.squeeze()
        
        metrics = {
            'filename': filename,
            'mae': self.calculate_mae(target, generated),
            'mse': self.calculate_mse(target, generated),
            'rmse': self.calculate_rmse(target, generated),
            'nrmse': self.calculate_nrmse(target, generated),
            'nmse': self.calculate_nmse(target, generated),
            'psnr': self.calculate_psnr(target, generated),
            'ssim': self.calculate_ssim(target, generated),
            'corr': self.calculate_correlation(target, generated)
        }
        
        return metrics
    
    def add_metrics(self, target, generated, filename):
        """Add metrics for a single image pair to the collection"""
        metrics = self.calculate_all_metrics(target, generated, filename)
        self.metrics_data.append(metrics)
        return metrics
    
    def save_metrics_to_csv(self, filename="LoHiResGAN_T1_all_slices_metrics.csv"):
        """Save all collected metrics to CSV in exact format"""
        if not self.metrics_data:
            print("No metrics data to save")
            return None
        
        # Create DataFrame with exact column order
        df = pd.DataFrame(self.metrics_data)
        column_order = ['filename', 'mae', 'mse', 'rmse', 'nrmse', 'nmse', 'psnr', 'ssim', 'corr']
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"ALL SLICES metrics saved to: {filename}")
        
        # Print summary statistics
        self.print_summary_statistics(df)
        
        return df
    
    def print_summary_statistics(self, df):
        """Print comprehensive summary statistics"""
        print("\nCOMPREHENSIVE METRICS SUMMARY (ALL SLICES):")
        print(f"Total slices evaluated: {len(df)}")
        print(f"Average MAE: {df['mae'].mean():.6f}")
        print(f"Average MSE: {df['mse'].mean():.6f}")
        print(f"Average RMSE: {df['rmse'].mean():.6f}")
        print(f"Average NRMSE: {df['nrmse'].mean():.6f}")
        print(f"Average NMSE: {df['nmse'].mean():.6f}")
        print(f"Average PSNR: {df['psnr'].mean():.6f}")
        print(f"Average SSIM: {df['ssim'].mean():.6f}")
        print(f"Average Correlation: {df['corr'].mean():.6f}")
    
    def get_metrics_count(self):
        """Get the number of metrics collected"""
        return len(self.metrics_data)
    
    def clear_metrics(self):
        """Clear all collected metrics"""
        self.metrics_data = []
    
    def get_latest_metrics(self):
        """Get the most recently added metrics"""
        if self.metrics_data:
            return self.metrics_data[-1]
        return None

if __name__ == "__main__":
    # Test the metrics calculator
    print("Testing ComprehensiveMetricsCalculator...")
    
    # Create dummy data for testing
    target = np.random.rand(256, 256)
    generated = target + np.random.normal(0, 0.1, target.shape)  # Add some noise
    
    calculator = ComprehensiveMetricsCalculator()
    metrics = calculator.calculate_all_metrics(target, generated, "test_image.nii.gz")
    
    print("Test metrics:")
    for key, value in metrics.items():
        if key != 'filename':
            print(f"  {key}: {value:.6f}")
    
    print("✅ Metrics calculator test completed successfully!")