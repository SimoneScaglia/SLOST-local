import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

# Base directory containing all the test size directories
base_dir = "../results/var_testset_experiments"

# Define the test sizes we're looking for (in order)
test_sizes = [5000, 10000, 15000, 20000, 25000, 30000]

# Lists to store our results
mean_aucs = []
mean_losses = []
actual_sizes = []

# Process each test size directory
for size in test_sizes:
    dir_name = f"5_0_testsize{size}"
    file_path = os.path.join(base_dir, dir_name, "swarm_results.csv")
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            mean_auc = df['auc'].mean()
            mean_loss = df['loss'].mean()
            
            mean_aucs.append(mean_auc)
            mean_losses.append(mean_loss)
            actual_sizes.append(size)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Create dual y-axis plot
if mean_aucs and mean_losses:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot AUC on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Testset Size')
    ax1.set_ylabel('AUC', color=color)
    line1 = ax1.plot(actual_sizes, mean_aucs, 'o-', color=color, linewidth=2, markersize=8, label='AUC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot Loss on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    line2 = ax2.plot(actual_sizes, mean_losses, 's-', color=color, linewidth=2, markersize=8, label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    plt.title('AUC and Loss vs Testset Size')
    plt.tight_layout()
    
    out_dir = 'plots_results/plot_linegraph_testsize'
    os.makedirs(out_dir, exist_ok=True)

    fig.savefig(f'{out_dir}/testset_size_dual_axis.png', dpi=300, bbox_inches='tight')
