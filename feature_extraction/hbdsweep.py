#!/usr/bin/env python3
"""GPU HDBSCAN Sweep -- adaptive parameter search.

Runs both 'eom' and 'leaf' methods with adaptive parameter updates.
Logs text output to results/hbdsweep_<name>.txt.

Usage:
    python -m feature_extraction.hbdsweep --base_dir ./feature_extraction/results/patch_coords
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
import cupy as cp
from cuml.cluster import HDBSCAN

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SAMPLE_SIZE = None        # set small int for test, None for full dataset
RANDOM_SEED = 42
TARGET_MIN, TARGET_MAX = 10, 35
IDEAL_CENTER = 25
MAX_MCS = 12000           # safety cap

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def make_grid(n_rows):
    """Initial sweep grid based on dataset size."""
    if n_rows >= 2_000_000:  # 8x8 patches
        mcs = [600, 1000, 1500, 2000]
        ms  = [10, 20, 30]
    else:                    # 16x16
        mcs = [200, 300, 400, 600]
        ms  = [5, 10, 20]
    return {"min_cluster_size": mcs, "min_samples": ms}

def cluster_score(nc):
    """Reward clusters near 25; penalize extremes."""
    if nc < TARGET_MIN or nc > 100:
        return 0
    return float(np.exp(-0.5 * ((nc - IDEAL_CENTER) / 10) ** 2))

def run_grid(X_gpu, grid, method="eom"):
    """Run HDBSCAN sweeps for each grid combination."""
    results = []
    total = len(grid["min_cluster_size"]) * len(grid["min_samples"])
    print(f"Running {total} configurations ({method})...", flush=True)

    for i, mcs in enumerate(grid["min_cluster_size"], 1):
        for ms in grid["min_samples"]:
            start = time.time()
            hdb = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_method=method,
            )
            labels_gpu = hdb.fit_predict(X_gpu)
            cp.cuda.Stream.null.synchronize()
            labels = cp.asnumpy(labels_gpu)

            n_clusters = len(np.unique(labels[labels != -1]))
            frac_noise = (labels == -1).sum() / len(labels)
            score = cluster_score(n_clusters)
            elapsed = (time.time() - start) / 60

            results.append({
                "method": method,
                "min_cluster_size": mcs,
                "min_samples": ms,
                "clusters": n_clusters,
                "noise%": frac_noise * 100,
                "score": score,
                "time_min": elapsed,
            })
            print(f"({i}/{total}) mcs={mcs}, ms={ms}, {method} -> "
                  f"k={n_clusters}, noise={frac_noise:.1%}, t={elapsed:.1f}m", flush=True)
    return pd.DataFrame(results)

def adjust_grid(grid, results_df):
    """Adjust min_cluster_size adaptively."""
    median_k = results_df["clusters"].median()
    print(f"Median clusters: {median_k:.1f}", flush=True)

    if median_k < TARGET_MIN:
        new_mcs = [max(50, int(m / 1.5)) for m in grid["min_cluster_size"]]
        print(f"Too few clusters -> lowering min_cluster_size: {new_mcs}", flush=True)
    elif median_k > TARGET_MAX:
        new_mcs = [int(m * 1.5) for m in grid["min_cluster_size"]]
        print(f"Too many clusters -> increasing min_cluster_size: {new_mcs}", flush=True)
    else:
        print("Cluster count in target range -- done.", flush=True)
        return None
    new_mcs = sorted(set(max(50, min(MAX_MCS, x)) for x in new_mcs))
    return {"min_cluster_size": new_mcs, "min_samples": grid["min_samples"]}


def main():
    parser = argparse.ArgumentParser(description="GPU HDBSCAN parameter sweep")
    parser.add_argument("--base_dir", required=True, help="Base directory containing _red.parquet files")
    parser.add_argument("--dataset", default="p8_s8", choices=["p8_s8", "p16_s16", "both"],
                        help="Which dataset(s) to sweep")
    args = parser.parse_args()

    BASE = args.base_dir

    all_datasets = {
        "p8_s8": os.path.join(BASE, "patch_coords_AMIL_ATR_8_4_0_fold0_fold0_p995_8x8_red.parquet"),
        "p16_s16": os.path.join(BASE, "patch_coords_AMIL_ATR_16_8_0_fold0_fold0_p995_16x16_red.parquet"),
    }

    if args.dataset == "both":
        DATASETS = [{"name": k, "path": v} for k, v in all_datasets.items()]
    else:
        DATASETS = [{"name": args.dataset, "path": all_datasets[args.dataset]}]

    os.makedirs("results", exist_ok=True)

    for ds in DATASETS:
        name, path = ds["name"], ds["path"]
        if not os.path.exists(path):
            print(f"Skipping {name}: file not found at {path}")
            continue

        logfile = f"results/hbdsweep_{name}.txt"
        with open(logfile, "w") as f:
            def logprint(*args):
                msg = " ".join(str(a) for a in args)
                print(msg, flush=True)
                f.write(msg + "\n")
                f.flush()

            logprint(f"\n===== Dataset: {name} =====")
            df = pd.read_parquet(path, columns=["umap_x", "umap_y"])
            N = len(df)
            logprint(f"Loaded {N:,} rows")

            if SAMPLE_SIZE is not None and N > SAMPLE_SIZE:
                df = df.sample(SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
                logprint(f"Using subset of {len(df):,} samples")
            else:
                logprint("Using full dataset")

            X_gpu = cp.asarray(df[["umap_x", "umap_y"]].to_numpy(), dtype=cp.float32)
            grid = make_grid(len(df))

            for attempt in range(1, 4):
                logprint(f"\n--- Sweep round {attempt} (method=eom) ---")
                results_eom = run_grid(X_gpu, grid, method="eom")
                logprint(results_eom.to_string(index=False))

                logprint(f"\n--- Sweep round {attempt} (method=leaf) ---")
                results_leaf = run_grid(X_gpu, grid, method="leaf")
                logprint(results_leaf.to_string(index=False))

                results_df = pd.concat([results_eom, results_leaf], ignore_index=True)
                new_grid = adjust_grid(grid, results_df)
                if new_grid is None:
                    break
                grid = new_grid

            best = results_df.sort_values(["score", "noise%"], ascending=[False, True]).iloc[0]
            logprint(f"\nFinal recommended for {name}: "
                     f"method={best.method}, "
                     f"min_cluster_size={best.min_cluster_size}, "
                     f"min_samples={best.min_samples}, "
                     f"{best.clusters} clusters, "
                     f"{best['noise%']:.1f}% noise, score={best.score:.3f}")

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
