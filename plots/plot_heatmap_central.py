import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

def build_directories(base_dir, learning_rates, batch_sizes, nodes):
    directories = []

    for lr in learning_rates:
        lr_str_5 = f"{lr:.5f}"

        for bs in batch_sizes:
            dir_5 = f"{nodes}_0_lr{lr_str_5}_bs{bs}"
            full_5 = os.path.join(base_dir, dir_5)

            if os.path.exists(full_5):
                directories.append(full_5)
                continue

            lr_str_4 = f"{lr:.4f}"
            dir_4 = f"{nodes}_0_lr{lr_str_4}_bs{bs}"
            full_4 = os.path.join(base_dir, dir_4)

            if os.path.exists(full_4):
                directories.append(full_4)
                continue

    return directories

def plot_heatmap_central(res_dir, rows, nodes):
    # Define the directories
    learning_rates = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    directories = build_directories(res_dir, learning_rates, batch_sizes, nodes)

    def parse_lr_from_dir(dir_path):
        """Extract learning rate from directory name"""
        dir_name = os.path.basename(dir_path)
        lr_part = dir_name.split('lr')[-1].split('_')[0]
        return float(lr_part)

    def parse_bs_from_dir(dir_path):
        """Extract batch size from directory name"""
        dir_name = os.path.basename(dir_path)
        bs_part = dir_name.split('bs')[-1]
        return int(bs_part)

    # Initialize dictionaries to store results
    auc_results = {bs: {lr: 0 for lr in learning_rates} for bs in batch_sizes}
    loss_results = {bs: {lr: 0 for lr in learning_rates} for bs in batch_sizes}

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
                # Store results
                if bs in auc_results and lr in auc_results[bs]:
                    auc_results[bs][lr] = df['auc'].mean()
                    loss_results[bs][lr] = df['loss'].mean()
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
    plt.figure(figsize=(28, 14))
    heatmap = sns.heatmap(auc_matrix, 
                xticklabels=[f"{lr:.5f}" for lr in learning_rates],
                yticklabels=batch_sizes,
                annot=True, 
                annot_kws={'size': 20, 'weight': 'normal'},
                fmt=".5f",
                cmap="RdYlGn",
                cbar_kws={'label': 'AUC'})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=18)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=18)
    plt.title(f'AUC Heatmap: Centralized-Node ({rows} rows)')
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    plt.tight_layout()
    plt.savefig(f'plots_results/plot_heatmap_central/auc_heatmap_{nodes}_{rows}.png', dpi=300, bbox_inches='tight')

    # Create Loss heatmap
    plt.figure(figsize=(28, 14))
    heatmap = sns.heatmap(loss_matrix, 
                xticklabels=[f"{lr:.5f}" for lr in learning_rates],
                yticklabels=batch_sizes,
                annot=True, 
                annot_kws={'size': 20, 'weight': 'normal'},
                fmt=".5f",
                cmap="RdYlGn_r",  # Reversed colormap for loss (red = bad/high loss)
                cbar_kws={'label': 'Loss'})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=18)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=18)
    plt.title(f'Loss Heatmap: Centralized-Node ({rows} rows)')
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    plt.tight_layout()
    plt.savefig(f'plots_results/plot_heatmap_central/loss_heatmap_{nodes}_{rows}.png', dpi=300, bbox_inches='tight')

def try_plot_heatmap_central(res_dir, rows, nodes):
    try:
        plot_heatmap_central(res_dir, rows, nodes)
    except Exception as e:
        print(f"An error occurred while plotting heatmap for {rows} rows/node: {e}")

if __name__ == "__main__":
    try_plot_heatmap_central(res_dir = '../results/heatmap_experiments_1000rows_5nodes', rows = 5000, nodes = 5)
    try_plot_heatmap_central(res_dir = '../results/heatmap_experiments_2000rows_5nodes', rows = 10000, nodes = 5)
    try_plot_heatmap_central(res_dir = '../results/heatmap_experiments_4000rows_5nodes', rows = 20000, nodes = 5)

    try_plot_heatmap_central(res_dir = '../results/heatmap_experiments_1000rows_2nodes', rows = 2000, nodes = 2)
    try_plot_heatmap_central(res_dir = '../results/heatmap_experiments_2000rows_2nodes', rows = 4000, nodes = 2)
    try_plot_heatmap_central(res_dir = '../results/heatmap_experiments_4000rows_2nodes', rows = 8000, nodes = 2)
