#!/usr/bin/env python3
"""Apply tuned HDBSCAN to reduced patch datasets.

Produces new Parquet files with added 'cluster' and 'probability' columns.

Usage:
    python -m feature_extraction.hdb_run --base_dir ./feature_extraction/results/patch_coords
"""

import argparse
import os
import pandas as pd
import cupy as cp
from cuml.cluster import HDBSCAN


def main():
    parser = argparse.ArgumentParser(description="Apply tuned HDBSCAN to reduced patch datasets")
    parser.add_argument("--base_dir", required=True, help="Base directory containing _red.parquet files")
    args = parser.parse_args()

    BASE = args.base_dir

    RUNS = [
        {
            "name": "p8_s8",
            "input": os.path.join(BASE, "patch_coords_AMIL_ATR_8_4_0_fold0_fold0_p995_8x8_red.parquet"),
            "output": os.path.join(BASE, "patch_coords_AMIL_ATR_8_4_0_fold0_fold0_p995_8x8_hdb.parquet"),
            "params": {"min_cluster_size": 4500, "min_samples": 30, "method": "eom"},
        },
        {
            "name": "p16_s16",
            "input": os.path.join(BASE, "patch_coords_AMIL_ATR_16_8_0_fold0_fold0_p995_16x16_red.parquet"),
            "output": os.path.join(BASE, "patch_coords_AMIL_ATR_16_8_0_fold0_fold0_p995_16x16_hdb.parquet"),
            "params": {"min_cluster_size": 1350, "min_samples": 5, "method": "eom"},
        },
    ]

    for run in RUNS:
        name = run["name"]
        in_path = run["input"]
        out_path = run["output"]
        params = run["params"]

        if not os.path.exists(in_path):
            print(f"Skipping {name}: input file not found at {in_path}")
            continue

        print(f"\n===== Processing {name} =====")
        print(f"Loading {in_path} ...")
        df = pd.read_parquet(in_path)
        print(f"-> Loaded {len(df):,} rows")

        # Prepare GPU data
        X_gpu = cp.asarray(df[["umap_x", "umap_y"]].to_numpy(), dtype=cp.float32)

        print(f"Running HDBSCAN (mcs={params['min_cluster_size']}, ms={params['min_samples']}, "
              f"method={params['method']}) ...")

        hdb = HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            cluster_selection_method=params["method"],
        )
        labels_gpu = hdb.fit_predict(X_gpu)
        probs_gpu = hdb.probabilities_

        cp.cuda.Stream.null.synchronize()
        df["cluster"] = cp.asnumpy(labels_gpu).astype("int32")
        df["probability"] = cp.asnumpy(probs_gpu).astype("float32")

        n_clusters = len(df["cluster"].unique()) - (1 if -1 in df["cluster"].unique() else 0)
        frac_noise = (df["cluster"] == -1).mean() * 100

        print(f"-> Found {n_clusters} clusters, {frac_noise:.1f}% noise")
        print(f"Saving enriched data -> {out_path}")
        df.to_parquet(out_path, index=False)
        print("Done.")

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":
    main()
