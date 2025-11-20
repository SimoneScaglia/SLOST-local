import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

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

def load_results(directories, csv_filename, learning_rates=None, batch_sizes=None):
    """Load results from directories with given CSV filename"""
    auc_results = {bs: {lr: 0 for lr in learning_rates} for bs in batch_sizes}
    loss_results = {bs: {lr: 0 for lr in learning_rates} for bs in batch_sizes}
    
    for directory in directories:
        csv_path = os.path.join(directory, csv_filename)
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                # Extract parameters
                lr = parse_lr_from_dir(directory)
                bs = parse_bs_from_dir(directory)

                if bs in auc_results and lr in auc_results[bs]:
                    auc_results[bs][lr] = df["auc"].mean()
                    loss_results[bs][lr] = df["loss"].mean()

            except Exception as e:
                print(f"Error processing {directory}: {e}")
        else:
            print(f"File not found: {csv_path}")

    return auc_results, loss_results

def build_directories(base_dir, learning_rates, batch_sizes):
    directories = []

    for lr in learning_rates:
        lr_str_5 = f"{lr:.5f}"

        for bs in batch_sizes:
            dir_5 = f"5_0_lr{lr_str_5}_bs{bs}"
            full_5 = os.path.join(base_dir, dir_5)

            if os.path.exists(full_5):
                directories.append(dir_5)
                continue

            lr_str_4 = f"{lr:.4f}"
            dir_4 = f"5_0_lr{lr_str_4}_bs{bs}"
            full_4 = os.path.join(base_dir, dir_4)

            if os.path.exists(full_4):
                directories.append(dir_4)
                continue

    return directories

def create_comparison_heatmap(swarm_res_dir, rows_per_node, metric='auc'):
    """
    Create a comparison heatmap showing swarm vs central results and their difference
    
    Parameters:
    - swarm_res_dir: directory containing swarm results
    - central_res_dir: directory containing central results  
    - rows_per_node: number of rows per node for swarm (total rows for central)
    - metric: 'auc' or 'loss'
    """
    
    learning_rates = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    directories = build_directories(swarm_res_dir, learning_rates, batch_sizes)
    
    # Build full paths
    swarm_dirs = [os.path.join(swarm_res_dir, d) for d in directories]
    central_dirs = [os.path.join(swarm_res_dir, d) for d in directories]
    
    # Load results
    swarm_auc, swarm_loss = load_results(swarm_dirs, 'swarm_results.csv', learning_rates, batch_sizes)
    central_auc, central_loss = load_results(central_dirs, 'central_results.csv', learning_rates, batch_sizes)
    
    # Select metric
    if metric == 'auc':
        swarm_matrix = swarm_auc
        central_matrix = central_auc
        metric_title = 'AUC'
    else:
        swarm_matrix = swarm_loss
        central_matrix = central_loss
        metric_title = 'Loss'
    
    # Create matrices
    batch_sizes = sorted(list(swarm_matrix.keys()))
    learning_rates = sorted(list(next(iter(swarm_matrix.values())).keys()))
    
    swarm_values = np.zeros((len(batch_sizes), len(learning_rates)))
    central_values = np.zeros((len(batch_sizes), len(learning_rates)))
    
    for i, bs in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):
            swarm_values[i, j] = swarm_matrix[bs][lr]
            central_values[i, j] = central_matrix[bs][lr]
    
    # Calculate difference
    if metric == 'auc':
        difference = swarm_values - central_values
        diff_label = 'Swarm - Central'
        cmap = 'RdYlGn'  # Green = swarm better, Red = central better
    else:  # loss
        difference = central_values - swarm_values  # Positive = swarm better (lower loss)
        diff_label = 'Central - Swarm\n(Positive = Swarm better)'
        cmap = 'RdYlGn_r'  # Reversed: Green = swarm better, Red = central better
    
    # Create output directory
    out_dir = 'plots_results/plot_comparison_heatmap'
    os.makedirs(out_dir, exist_ok=True)
    
    # Create the comparison heatmap
    plt.figure(figsize=(28, 14))
    
    # Create annotated matrix for cell text
    annotation_matrix = np.empty_like(difference, dtype=object)
    for i in range(len(batch_sizes)):
        for j in range(len(learning_rates)):
            swarm_val = swarm_values[i, j]
            central_val = central_values[i, j]
            diff_val = difference[i, j]
            
            annotation_matrix[i, j] = f"S:{swarm_val:.4f}\nC:{central_val:.4f}\nΔ:{diff_val:+.4f}"
    
    # Plot heatmap
    heatmap = sns.heatmap(difference,
                        xticklabels=[f"{lr:.5f}" for lr in learning_rates],
                        yticklabels=batch_sizes,
                        annot=annotation_matrix,
                        fmt="",
                        cmap=cmap,
                        cbar_kws={'label': diff_label},
                        linewidths=0.5,
                        linecolor='white',
                        annot_kws={'size': 18, 'weight': 'normal', 'color': 'black'},
                        vmin=-0.05 if metric == 'auc' else -0.05,
                        vmax=0.05 if metric == 'auc' else 0.05)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=18)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=18)
    
    plt.title(f'{metric_title} Comparison: Swarm vs Central\n'
            f'Swarm: 5 nodes × {rows_per_node} rows/node = {5*rows_per_node} total rows\n'
            f'Central: {5*rows_per_node} total rows\n'
            f'Green: Swarm better, Red: Central better')
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Save plot
    filename = f'{out_dir}/comparison_heatmap_{metric}_{rows_per_node}rows.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison heatmap saved: {filename}")

def try_plot_comparison(swarm_res_dir, rows_per_node):
    """Wrapper function with error handling"""
    try:
        print(f"Processing comparison for {rows_per_node} rows/node...")
        create_comparison_heatmap(swarm_res_dir, rows_per_node, metric='auc')
        create_comparison_heatmap(swarm_res_dir, rows_per_node, metric='loss')
    except Exception as e:
        print(f"An error occurred while plotting comparison for {rows_per_node} rows/node: {e}")

if __name__ == "__main__":
    # Define the mapping between swarm and central directories
    comparisons = [
        {
            'swarm': '../results/heatmap_experiments_1000rows_5nodes',
            'rows': 1000
        },
        {
            'swarm': '../results/heatmap_experiments_2000rows_5nodes', 
            'rows': 2000
        },
        {
            'swarm': '../results/heatmap_experiments_4000rows_5nodes',
            'rows': 4000
        }
    ]
    
    for comp in comparisons:
        try_plot_comparison(comp['swarm'], comp['rows'])