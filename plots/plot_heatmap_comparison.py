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

def load_results(directories, csv_filename):
    """Load results from directories with given CSV filename"""
    auc_results = {}
    loss_results = {}
    
    for directory in directories:
        csv_path = os.path.join(directory, csv_filename)
        
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
    
    return auc_results, loss_results

def create_comparison_heatmap(swarm_res_dir, central_res_dir, rows_per_node, metric='auc'):
    """
    Create a comparison heatmap showing swarm vs central results and their difference
    
    Parameters:
    - swarm_res_dir: directory containing swarm results
    - central_res_dir: directory containing central results  
    - rows_per_node: number of rows per node for swarm (total rows for central)
    - metric: 'auc' or 'loss'
    """
    
    # Define the directories structure (same for both)
    directories = [
        f"5_0_lr0.0010_bs32",
        f"5_0_lr0.0010_bs64",
        f"5_0_lr0.0010_bs128",
        f"5_0_lr0.0025_bs32",
        f"5_0_lr0.0025_bs64",
        f"5_0_lr0.0025_bs128",
        f"5_0_lr0.0050_bs32",
        f"5_0_lr0.0050_bs64",
        f"5_0_lr0.0050_bs128",
        f"5_0_lr0.0100_bs32",
        f"5_0_lr0.0100_bs64",
        f"5_0_lr0.0100_bs128",
        f"5_0_lr0.0250_bs32",
        f"5_0_lr0.0250_bs64",
        f"5_0_lr0.0250_bs128",
        f"5_0_lr0.0500_bs32",
        f"5_0_lr0.0500_bs64",
        f"5_0_lr0.0500_bs128",
        f"5_0_lr0.0750_bs32",
        f"5_0_lr0.0750_bs64",
        f"5_0_lr0.0750_bs128"
    ]
    
    # Build full paths
    swarm_dirs = [os.path.join(swarm_res_dir, d) for d in directories]
    central_dirs = [os.path.join(central_res_dir, d) for d in directories]
    
    # Load results
    swarm_auc, swarm_loss = load_results(swarm_dirs, 'swarm_results.csv')
    central_auc, central_loss = load_results(central_dirs, 'central_results.csv')
    
    # Select metric
    if metric == 'auc':
        swarm_matrix = swarm_auc
        central_matrix = central_auc
        metric_title = 'AUC'
        # For AUC, higher is better, so difference = swarm - central
        # Positive difference means swarm performs better
    else:  # loss
        swarm_matrix = swarm_loss
        central_matrix = central_loss
        metric_title = 'Loss'
        # For loss, lower is better, so difference = central - swarm
        # Positive difference means swarm performs better (lower loss)
    
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
    plt.figure(figsize=(16, 12))
    
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
                        xticklabels=[f"{lr:.4f}" for lr in learning_rates],
                        yticklabels=batch_sizes,
                        annot=annotation_matrix,
                        fmt='',
                        cmap=cmap,
                        cbar_kws={'label': diff_label},
                        linewidths=0.5,
                        linecolor='white',
                        annot_kws={'size': 16, 'weight': 'normal', 'color': 'black'},
                        vmin=-0.05 if metric == 'auc' else -0.05,
                        vmax=0.05 if metric == 'auc' else 0.05)
    
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

def try_plot_comparison(swarm_res_dir, central_res_dir, rows_per_node):
    """Wrapper function with error handling"""
    try:
        print(f"Processing comparison for {rows_per_node} rows/node...")
        create_comparison_heatmap(swarm_res_dir, central_res_dir, rows_per_node, metric='auc')
        create_comparison_heatmap(swarm_res_dir, central_res_dir, rows_per_node, metric='loss')
    except Exception as e:
        print(f"An error occurred while plotting comparison for {rows_per_node} rows/node: {e}")

if __name__ == "__main__":
    # Define the mapping between swarm and central directories
    comparisons = [
        {
            'swarm': '../results/heatmap_experiments_1000rows_5nodes',
            'central': '../results/heatmap_experiments_central_1000rows_5nodes',
            'rows': 1000
        },
        {
            'swarm': '../results/heatmap_experiments_2000rows_5nodes', 
            'central': '../results/heatmap_experiments_central_2000rows_5nodes',
            'rows': 2000
        },
        {
            'swarm': '../results/heatmap_experiments_4000rows_5nodes',
            'central': '../results/heatmap_experiments_central_4000rows_5nodes', 
            'rows': 4000
        }
    ]
    
    for comp in comparisons:
        try_plot_comparison(comp['swarm'], comp['central'], comp['rows'])