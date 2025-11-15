#!/usr/bin/env python3
import os
import sys
import gc
import pickle
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
import multiprocessing as mp

os.chdir(Path(__file__).resolve().parent)

# seeds
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

LABEL_COL = "is_sepsis"
NODES = 10              ## TODO: fissare nodi
DATA_DIR = Path(f"../../datasets/{NODES}nodes")
TEST_FILE = Path(f"../../datasets/{NODES}nodes/test.csv")

# parametri fissi
EPOCHS = 25             ## TODO: fissare parametri
BATCH_SIZE = 64        ## TODO: fissare parametri
LEARNING_RATE = 0.001  ## TODO: fissare parametri

# fixed output values
OUTPUT_USER = "local"
OUTPUT_SPLITS = 100
OUTPUT_COLUMNS = ["datetime", "user", "splits", "loss", "auc", "auprc", "accuracy", "precision", "recall", "iteration"]

# parallelismo: numero di processi figli. Se None -> min(4, cpu_count())
MAX_PROCESSES: Optional[int] = None

# salva la cache su disco per riutilizzo
CACHE_PATH = Path("local_train_eval_cache.pkl")


# ==========================
# DEFINIZIONE MODELLO FCN
# ==========================
def build_fcn(input_dim: int) -> keras.Model:
    """Costruisce una Fully Connected Network come in mimic_nets.py"""
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=42)
    bias_initializer = tf.keras.initializers.Zeros()

    model = keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(input_dim,), kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(16, activation="relu", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        keras.layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    ])
    return model


def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


def get_metrics():
    return [
        AUC(name='auc', curve='ROC', num_thresholds=1000),
        AUC(name='auprc', curve='PR', num_thresholds=1000),
        BinaryAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall')
    ]


# ==========================
# FUNZIONI DI UTILITÀ
# ==========================
def prepare_xy(df: pd.DataFrame, feature_columns=None):
    if LABEL_COL not in df.columns:
        raise KeyError(f"Label column '{LABEL_COL}' non trovata nel dataframe.")
    if feature_columns is None:
        X_cols = [c for c in df.columns if c != LABEL_COL]
    else:
        X_cols = feature_columns
    missing = [c for c in X_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    X = df[X_cols].astype(np.float32).to_numpy()
    y = df[LABEL_COL].astype(np.float32).to_numpy().reshape(-1, 1)
    return X, y, X_cols


def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def append_result_csv(path: Path, row: dict):
    full = {col: row.get(col, None) for col in OUTPUT_COLUMNS}
    df_row = pd.DataFrame([full], columns=OUTPUT_COLUMNS)
    if path.exists():
        df_row.to_csv(path, mode='a', header=False, index=False)
    else:
        ensure_dir(path.parent)
        df_row.to_csv(path, mode='w', header=True, index=False)


def current_datetime_rome_iso():
    return datetime.now(ZoneInfo("Europe/Rome")).isoformat()


# ==========================
# TRAIN + EVAL
# ==========================
def train_and_eval(node_idx: int, iteration_y: int, epochs: int, batch_size: int, verbose=0):
    # ogni processo figlio riallinea i seed per riproducibilità
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    print(f"=== computing node={node_idx}, iteration={iteration_y} ===")
    train_file = DATA_DIR / f"node{node_idx}_{iteration_y}.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"File mancante: {train_file}")

    df_train = pd.read_csv(train_file)
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file mancante: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE)

    # align features
    X_train, y_train, feature_cols = prepare_xy(df_train, feature_columns=None)
    X_test, y_test, _ = prepare_xy(df_test, feature_columns=feature_cols)

    input_dim = X_train.shape[1]
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}, input_dim={input_dim}")

    # build and train
    model = build_fcn(input_dim=input_dim)
    model.compile(
        optimizer=get_optimizer(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=get_metrics()
    )

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)

    eval_results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)

    # cleanup
    tf.keras.backend.clear_session()
    del model
    del df_train, df_test, X_train, y_train, X_test, y_test
    gc.collect()

    return eval_results


# wrapper robusto usato dal pool: ritorna una tupla (node, iteration, results or None, error_message)
def safe_worker(args: Tuple[int, int, int, int]):
    node_idx, iteration_y, epochs, batch_size = args
    try:
        res = train_and_eval(node_idx, iteration_y, epochs, batch_size)
        return (node_idx, iteration_y, res, None)
    except FileNotFoundError as e:
        # file mancante: non è un errore fatale per lo script, semplicemente saltiamo
        return (node_idx, iteration_y, None, str(e))
    except Exception as e:
        # segnaliamo l'errore ma continuiamo
        return (node_idx, iteration_y, None, f"EXCEPTION: {e}")


# ---------------------- main -------------------------------------------------
def discover_existing_pairs(data_dir: Path) -> Dict[Tuple[int, int], Path]:
    pairs = {}
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory dati non trovata: {data_dir}")
    for p in data_dir.glob("node*_*.csv"):
        name = p.stem  # node{idx}_{y}
        if "_" not in name:
            continue
        try:
            node_part, iter_part = name.split("_")
            node_idx = int(node_part.replace("node", ""))
            iteration_y = int(iter_part)
            pairs[(node_idx, iteration_y)] = p
        except Exception:
            continue
    return pairs


def build_tasks(pairs_keys):
    tasks = []
    for (node_idx, iteration_y) in pairs_keys:
        tasks.append((node_idx, iteration_y, EPOCHS, BATCH_SIZE))
    return tasks


def main():
    config_nodes = list(range(2, NODES + 1))
    iterations = list(range(10))

    # scopri quali file esistono realmente
    existing = discover_existing_pairs(DATA_DIR)
    if not existing:
        print("Nessun file node*_*.csv trovato in", DATA_DIR)
        return

    # tasks da eseguire: tutte le coppie esistenti
    tasks = build_tasks(sorted(existing.keys()))
    print(f"Trovate {len(tasks)} coppie (node,iteration) da calcolare")

    # eventualmente carica cache da disco
    cache: Dict[Tuple[int, int], dict] = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            print(f"Caricata cache con {len(cache)} elementi da {CACHE_PATH}")
        except Exception:
            cache = {}

    # build lista di tasks che effettivamente mancano in cache
    tasks_to_run = [t for t in tasks if (t[0], t[1]) not in cache]
    print(f"Da calcolare: {len(tasks_to_run)} coppie (gli altri saranno riutilizzati dalla cache)")

    if tasks_to_run:
        procs = MAX_PROCESSES or min(12, mp.cpu_count())
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=procs) as pool:
            results = pool.map(safe_worker, tasks_to_run)

        # processa i risultati
        for node_idx, iteration_y, res, err in results:
            if res is not None:
                cache[(node_idx, iteration_y)] = res
            else:
                print(f"Saltata coppia node{node_idx}_{iteration_y}: {err}")

        # salva cache su disco
        try:
            ensure_dir(CACHE_PATH.parent)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)
            print(f"Cache salvata in {CACHE_PATH}")
        except Exception as e:
            print("Impossibile salvare la cache:", e)

    # ora per ogni configurazione copia le righe necessarie dalla cache
    for x in config_nodes:
        out_file = Path(f"../../results/{NODES}nodes_results/1host_{x}nodes/local_results/local_results.csv")
        # se esiste un file precedente lo rimettiamo da zero per coerenza (opzionale)
        if out_file.exists():
            print(f"Rimuovo file esistente {out_file} per riscrivere i risultati")
            out_file.unlink()

        for y in iterations:
            for node_idx in range(1, x + 1):
                key = (node_idx, y)
                if key not in cache:
                    # se non c'è nel cache, skip (è probabile che il file node... non esistesse)
                    continue
                results = cache[key]
                row = {
                    "datetime": current_datetime_rome_iso(),
                    "user": f"local_node{node_idx}",
                    "splits": OUTPUT_SPLITS,
                    'loss': results.get('loss'),
                    'auc': results.get('auc'),
                    'auprc': results.get('auprc'),
                    'accuracy': results.get('accuracy'),
                    'precision': results.get('precision'),
                    'recall': results.get('recall'),
                    "iteration": y
                }
                append_result_csv(out_file, row)
        print(f"Scritti risultati per configurazione 1host_{x}nodes -> {out_file}")

    # rimuovo la cache a fine esecuzione
    if CACHE_PATH.exists():
        try:
            CACHE_PATH.unlink()
            print(f"Rimossa cache temporanea {CACHE_PATH}")
        except Exception as e:
            print(f"Impossibile rimuovere la cache {CACHE_PATH}: {e}")
    print("Completato.")


if __name__ == "__main__":
    main()
