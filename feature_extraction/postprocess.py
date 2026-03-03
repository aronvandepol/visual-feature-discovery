#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU Reduction Script -- UMAP + t-SNE
-----------------------------------
Loads a parquet of image patch embeddings, normalizes them,
runs UMAP and t-SNE (both on GPU), and saves the results back
into a new parquet with `_red` appended to the original name.

If SAMPLE_SIZE is set to None, the full dataset is used.

Usage:
    python -m feature_extraction.postprocess --input path/to/patches.parquet
"""

import argparse
import os
import time
import cupy as cp
import numpy as np
import pandas as pd
from cuml.manifold import UMAP, TSNE


def main():
    parser = argparse.ArgumentParser(description="GPU UMAP + t-SNE reduction for patch embeddings")
    parser.add_argument("--input", required=True, help="Path to input parquet with embeddings")
    parser.add_argument("--output", default=None, help="Output path (default: input with _red suffix)")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of rows to sample (None=full)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    PARQUET_PATCHES_PATH = args.input
    SAMPLE_SIZE = args.sample_size
    RANDOM_SEED = args.seed
    OUT_PATH = args.output or PARQUET_PATCHES_PATH.replace(".parquet", "_red.parquet")

    # ------------------------------------------------------------
    # 1. Load parquet + embeddings
    # ------------------------------------------------------------
    t0 = time.time()
    print(f"Loading dataset from: {PARQUET_PATCHES_PATH}")

    df = pd.read_parquet(PARQUET_PATCHES_PATH)
    assert "embedding" in df.columns, "Missing 'embedding' column in parquet"
    X = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
    print(f"Loaded {len(df):,} rows | embedding dim = {X.shape[1]}")

    t_load_end = time.time()

    # ------------------------------------------------------------
    # 2. Normalize + optional subsample
    # ------------------------------------------------------------
    t_norm = time.time()
    X_gpu_full = cp.asarray(X, dtype=cp.float32)
    norms = cp.linalg.norm(X_gpu_full, axis=1, keepdims=True)
    X_gpu_full /= (norms + 1e-8)

    N_total = X_gpu_full.shape[0]

    if SAMPLE_SIZE is not None and N_total > SAMPLE_SIZE:
        cp.random.seed(RANDOM_SEED)
        idx_gpu = cp.random.permutation(N_total)[:SAMPLE_SIZE]
        idx_cpu = cp.asnumpy(idx_gpu)
        X_gpu = X_gpu_full[idx_gpu]
        df_vis = df.iloc[idx_cpu].reset_index(drop=True)
        print(f"-> Using {len(df_vis):,} sampled rows (of {N_total:,})")
    else:
        X_gpu = X_gpu_full
        df_vis = df.reset_index(drop=True)
        print(f"-> Using full dataset ({N_total:,} rows)")

    t_norm_end = time.time()

    # ------------------------------------------------------------
    # 3. GPU UMAP
    # ------------------------------------------------------------
    t_umap = time.time()
    print("Running UMAP...")
    umap_gpu = UMAP(
        n_neighbors=30,
        n_components=2,
        min_dist=0.05,
        init="spectral",
        random_state=RANDOM_SEED,
        verbose=True,
    )
    X_umap_gpu = umap_gpu.fit_transform(X_gpu)
    cp.cuda.Stream.null.synchronize()
    t_umap_end = time.time()

    # ------------------------------------------------------------
    # 4. GPU t-SNE
    # ------------------------------------------------------------
    t_tsne = time.time()
    print("Running t-SNE...")
    tsne_gpu = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=RANDOM_SEED,
        verbose=True,
    )
    X_tsne_gpu = tsne_gpu.fit_transform(X_gpu)
    cp.cuda.Stream.null.synchronize()
    t_tsne_end = time.time()

    # ------------------------------------------------------------
    # 5. Attach results to dataframe
    # ------------------------------------------------------------
    X_umap = cp.asnumpy(X_umap_gpu)
    X_tsne = cp.asnumpy(X_tsne_gpu)

    df_vis["umap_x"] = X_umap[:, 0]
    df_vis["umap_y"] = X_umap[:, 1]
    df_vis["tsne_x"] = X_tsne[:, 0]
    df_vis["tsne_y"] = X_tsne[:, 1]

    # ------------------------------------------------------------
    # 6. Save with _red suffix
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    df_vis.to_parquet(OUT_PATH, index=False)
    print(f"Saved reduced dataset -> {OUT_PATH}")

    # ------------------------------------------------------------
    # 7. Timing summary
    # ------------------------------------------------------------
    total_time = time.time() - t0
    print("\nTiming summary")
    print(f"Load parquet:         {t_load_end - t0:>6.2f}s")
    print(f"Normalize/subsample:  {t_norm_end - t_norm:>6.2f}s")
    print(f"UMAP:                 {t_umap_end - t_umap:>6.2f}s")
    print(f"t-SNE:                {t_tsne_end - t_tsne:>6.2f}s")
    print(f"Total runtime:        {total_time/60:>6.2f} min")
    print("Done.")


if __name__ == "__main__":
    main()
