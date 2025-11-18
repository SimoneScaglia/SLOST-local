import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

def plot_heatmap_central(res_dir, rows):
    # Define the directories
    directories = [
        f"{res_dir}/5_0_lr0.0010_bs32",
        f"{res_dir}/5_0_lr0.0010_bs64",
        f"{res_dir}/5_0_lr0.0010_bs128",
        f"{res_dir}/5_0_lr0.0025_bs32",
        f"{res_dir}/5_0_lr0.0025_bs64",
        f"{res_dir}/5_0_lr0.0025_bs128",
        f"{res_dir}/5_0_lr0.0050_bs32",
        f"{res_dir}/5_0_lr0.0050_bs64",
        f"{res_dir}/5_0_lr0.0050_bs128",
        f"{res_dir}/5_0_lr0.0100_bs32",
        f"{res_dir}/5_0_lr0.0100_bs64",
        f"{res_dir}/5_0_lr0.0100_bs128",
        f"{res_dir}/5_0_lr0.0250_bs32",
        f"{res_dir}/5_0_lr0.0250_bs64",
        f"{res_dir}/5_0_lr0.0250_bs128",
        f"{res_dir}/5_0_lr0.0500_bs32",
        f"{res_dir}/5_0_lr0.0500_bs64",
        f"{res_dir}/5_0_lr0.0500_bs128",
        f"{res_dir}/5_0_lr0.0750_bs32",
        f"{res_dir}/5_0_lr0.0750_bs64",
        f"{res_dir}/5_0_lr0.0750_bs128"
    ]

    def parse_lr_from_dir(dir_path):
        """Extract learning rate from directory name"""
        dir_name = os.path.basename(dir_path)
        lr_part = dir_name.split('lr')[-1].split('_')[0]
        # Convert string like '0-0010' to float 0.001
        lr_str = lr_part.replace('-', '.')
        return float(lr_str)

    def parse_bs_from_dir(dir_path):
        """Extract batch size from directory name"""
        dir_name = os.path.basename(dir_path)
        bs_part = dir_name.split('bs')[-1]
        return int(bs_part)

    # Initialize dictionaries to store results
    auc_results = {}
    loss_results = {}

    # Process each directory
    for directory in directories:
        csv_path = os.path.join(directory, 'central_results.csv')
        
        if os.path.exists(csv_path):
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)
                
                # Extract parameters from directory name
                lr = parse_lr_from_dir(directory)
                bs = parse_bs_from_dir(directory)
                
                # Calculate mean values
                mean_auc = df['auc'].mean()
                mean_loss = df['loss'].mean()
                
                # Store results
                if bs not in auc_results:
                    auc_results[bs] = {}
                    loss_results[bs] = {}
                
                auc_results[bs][lr] = mean_auc
                loss_results[bs][lr] = mean_loss
            except Exception as e:
                print(f"Error processing {directory}: {e}")
        else:
            print(f"File not found: {csv_path}")

    # Create DataFrames for heatmaps
    batch_sizes = sorted(list(auc_results.keys()))
    learning_rates = sorted(list(next(iter(auc_results.values())).keys()))

    auc_matrix = np.zeros((len(batch_sizes), len(learning_rates)))
    loss_matrix = np.zeros((len(batch_sizes), len(learning_rates)))

    for i, bs in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):
            auc_matrix[i, j] = auc_results[bs][lr]
            loss_matrix[i, j] = loss_results[bs][lr]

    out_dir = 'plots_results/plot_heatmap_central'
    os.makedirs(out_dir, exist_ok=True)

    # Create AUC heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(auc_matrix, 
                xticklabels=[f"{lr:.4f}" for lr in learning_rates],
                yticklabels=batch_sizes,
                annot=True, 
                fmt=".4f",
                cmap="RdYlGn",
                cbar_kws={'label': 'AUC'})
    plt.title(f'AUC Heatmap: Centralized-Node ({rows} rows)')
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    plt.tight_layout()
    plt.savefig(f'plots_results/plot_heatmap_central/auc_heatmap_{rows}.png', dpi=300, bbox_inches='tight')

    # Create Loss heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(loss_matrix, 
                xticklabels=[f"{lr:.4f}" for lr in learning_rates],
                yticklabels=batch_sizes,
                annot=True, 
                fmt=".4f",
                cmap="RdYlGn_r",  # Reversed colormap for loss (red = bad/high loss)
                cbar_kws={'label': 'Loss'})
    plt.title(f'Loss Heatmap: Centralized-Node ({rows} rows)')
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    plt.tight_layout()
    plt.savefig(f'plots_results/plot_heatmap_central/loss_heatmap_{rows}.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"AUC Range: {auc_matrix.min():.4f} - {auc_matrix.max():.4f}")
    print(f"Loss Range: {loss_matrix.min():.4f} - {loss_matrix.max():.4f}")

    # Find best configurations
    best_auc_idx = np.unravel_index(np.argmax(auc_matrix), auc_matrix.shape)
    best_loss_idx = np.unravel_index(np.argmin(loss_matrix), loss_matrix.shape)

    print(f"\nBest AUC: {auc_matrix[best_auc_idx]:.4f} at BS={batch_sizes[best_auc_idx[0]]}, LR={learning_rates[best_auc_idx[1]]:.4f}")
    print(f"Best Loss: {loss_matrix[best_loss_idx]:.4f} at BS={batch_sizes[best_loss_idx[0]]}, LR={learning_rates[best_loss_idx[1]]:.4f}")

def try_plot_heatmap_central(res_dir, rows):
    try:
        plot_heatmap_central(res_dir, rows)
    except Exception as e:
        print(f"An error occurred while plotting heatmap for {rows} rows/node: {e}")

if __name__ == "__main__":
    try_plot_heatmap_central(res_dir = '../results/heatmap_experiments_central_1000rows_5nodes', rows = 5000)
    try_plot_heatmap_central(res_dir = '../results/heatmap_experiments_central_2000rows_5nodes', rows = 10000)
    try_plot_heatmap_central(res_dir = '../results/heatmap_experiments_central_4000rows_5nodes', rows = 20000)
