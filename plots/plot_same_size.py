#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

def safe_stats_from_csv(path, col='auc'):
    """Se il file esiste e ha la colonna, ritorna (mean, min, max, count).
        Altrimenti ritorna (np.nan, np.nan, np.nan, 0)."""
    if not os.path.exists(path):
        return np.nan, np.nan, np.nan, 0
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Warning: impossibile leggere {path}: {e}")
        return np.nan, np.nan, np.nan, 0
    if col not in df.columns:
        print(f"Warning: colonna '{col}' non trovata in {path}")
        return np.nan, np.nan, np.nan, 0
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    vals = df.groupby("iteration")[col].mean().dropna()
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, 0
    return float(vals.mean()), float(vals.min()), float(vals.max()), int(len(vals))

def main(results_dir, out_file, show_local, show_annotations, nodes, metric, metric_min, metric_max):
    x_min, x_max = 1, nodes + 1
    xs = np.arange(x_min, x_max + 1)

    swarm_mean = np.full(len(xs), np.nan)
    swarm_min = np.full(len(xs), np.nan)
    swarm_max = np.full(len(xs), np.nan)
    swarm_count = np.zeros(len(xs), dtype=int)

    central_mean = np.full(len(xs), np.nan)
    central_min = np.full(len(xs), np.nan)
    central_max = np.full(len(xs), np.nan)
    central_count = np.zeros(len(xs), dtype=int)

    local_mean = np.full(len(xs), np.nan)
    local_min = np.full(len(xs), np.nan)
    local_max = np.full(len(xs), np.nan)
    local_count = np.zeros(len(xs), dtype=int)

    for n in range(2, nodes + 1):
        folder_name = f"1host_{n}nodes"
        base = os.path.join(results_dir, folder_name)
        swarm_path = os.path.join(base, "swarm_results", "swarm_results.csv")
        central_path = os.path.join(base, "central_results", "central_results.csv")
        local_path = os.path.join(base, "local_results", "local_results.csv")

        s_mean, s_min, s_max, s_count = safe_stats_from_csv(swarm_path, col=metric)
        c_mean, c_min, c_max, c_count = safe_stats_from_csv(central_path, col=metric)
        l_mean, l_min, l_max, l_count = safe_stats_from_csv(local_path, col=metric)

        ix = n - x_min
        if 0 <= ix < len(xs):
            swarm_mean[ix] = s_mean
            swarm_min[ix] = s_min
            swarm_max[ix] = s_max
            swarm_count[ix] = s_count

            central_mean[ix] = c_mean
            central_min[ix] = c_min
            central_max[ix] = c_max
            central_count[ix] = c_count

            local_mean[ix] = l_mean
            local_min[ix] = l_min
            local_max[ix] = l_max
            local_count[ix] = l_count

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # marker size ridotta
    msize = 3

    swarm_line, = ax.plot(xs, swarm_mean, linestyle='-', marker='o', markersize=msize, label='Swarm (media)', linewidth=1.6)
    central_line, = ax.plot(xs, central_mean, linestyle='--', marker='s', markersize=msize, label='Central (media)', linewidth=1.6)
    if show_local:
        local_line, = ax.plot(xs, local_mean, linestyle=':', marker='v', markersize=msize, label='Local (media)', linewidth=1.6)

    # fill min-max per swarm (se presenti)
    valid_swarm = ~np.isnan(swarm_min) & ~np.isnan(swarm_max)
    if valid_swarm.any():
        ax.fill_between(xs, swarm_min, swarm_max, where=valid_swarm, alpha=0.15, interpolate=True, label='Swarm (min-max)', color=swarm_line.get_color())

    # fill min-max per central (se presenti)
    valid_central = ~np.isnan(central_min) & ~np.isnan(central_max)
    if valid_central.any():
        ax.fill_between(xs, central_min, central_max, where=valid_central, alpha=0.15, interpolate=True, label='Central (min-max)', color=central_line.get_color())

    # fill min-max per local (se presenti)
    valid_local = ~np.isnan(local_min) & ~np.isnan(local_max)
    if valid_local.any() and show_local:
        ax.fill_between(xs, local_min, local_max, where=valid_local, alpha=0.15, interpolate=True, label='Local (min-max)', color=local_line.get_color())

    # limiti e ticks
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(x_min, x_max + 1, 1))
    ax.set_xlabel("Numero di nodi")
    ax.set_ylim(metric_min, metric_max)
    ax.set_ylabel(metric)
    ax.set_title(f"Swarm vs Central (epochs = 25, batch size = 64, learning rate = 0.001, {int(40000/nodes)} righe x nodo)")
    ax.grid(axis='both', linestyle='--', alpha=0.5)

    ax.legend(loc='lower right')

    # annotazioni discrete sopra i punti (solo per i n con dati)
    if show_annotations:
        for i, x in enumerate(xs):
            if not np.isnan(swarm_mean[i]):
                ax.annotate(f"{swarm_mean[i]:.3f}", (x, swarm_mean[i]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=7, color=swarm_line.get_color())
            if not np.isnan(central_mean[i]):
                ax.annotate(f"{central_mean[i]:.3f}", (x, central_mean[i]), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=7, color=central_line.get_color())
            if not np.isnan(local_mean[i]) and show_local:
                ax.annotate(f"{local_mean[i]:.3f}", (x, local_mean[i]), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=7, color=local_line.get_color())

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Grafico salvato in: {out_file}")
    plt.close()

if __name__ == "__main__":
    data_dir = '../results'
    out_dir = 'plots_results/plot_same_size'
    os.makedirs(out_dir, exist_ok=True)

    for config in [10, 20, 40, 80]:
        for metric in ['auc', 'loss']:
            main(
                f"{data_dir}/{config}nodes_results",
                f"{out_dir}/results_auc_comparison_{config}nodes_{metric}.png",
                True, False, config, metric,
                0.8 if metric == 'auc' else 0.25,
                0.9 if metric == 'auc' else (0.8 if config == 80 else 0.4)
            )
