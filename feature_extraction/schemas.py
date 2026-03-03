#!/usr/bin/env python3
"""Schema helpers for patch extraction outputs."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def get_core_schema(save_embeddings: bool = True) -> pa.schema:
    """Return the core Arrow schema used for extracted patch coordinates."""
    fields = [
        pa.field("image_path", pa.string()),
        pa.field("printshop", pa.string()),
        pa.field("book", pa.string()),
        pa.field("book_id", pa.string()),
        pa.field("page_index", pa.int64()),
        pa.field("grid_h", pa.int32()),
        pa.field("grid_w", pa.int32()),
        pa.field("flat_idx", pa.int32()),
        pa.field("h_coord", pa.int32()),
        pa.field("w_coord", pa.int32()),
        pa.field("pixel_start_h", pa.int32()),
        pa.field("pixel_start_w", pa.int32()),
        pa.field("pixel_end_h", pa.int32()),
        pa.field("pixel_end_w", pa.int32()),
        pa.field("patch_size", pa.int16()),
        pa.field("stride_used", pa.int16()),
        pa.field("attention_weight", pa.float32()),
        pa.field("attention_weight_norm", pa.float32()),
        pa.field("percentile_used", pa.float32()),
        pa.field("model_experiment", pa.string()),
        pa.field("model_fold", pa.int8()),
        pa.field("publication_date", pa.string()),
        pa.field("publication_year", pa.int32()),
    ]
    if save_embeddings:
        fields.append(pa.field("embedding", pa.list_(pa.float16())))
    return pa.schema(fields)


def make_output_paths(
    output_dir: str,
    exp_name: str,
    fold: int,
    patch_size: int,
    stride: int,
    percentile: float,
    suffix: str = "",
    add_timestamp: bool = True,
) -> Tuple[str, str]:
    """Return temporary and final parquet file paths."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
    base = f"patch_coords_{exp_name}_fold{fold}_p{int(percentile*10)}_{patch_size}x{stride}{suffix}"
    if timestamp:
        base = f"{base}_{timestamp}"
    temp_path = os.path.join(output_dir, f"{base}.tmp.parquet")
    final_path = os.path.join(output_dir, f"{base}.parquet")
    return temp_path, final_path


def open_writer(temp_path: str, schema: pa.schema) -> pq.ParquetWriter:
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    return pq.ParquetWriter(temp_path, schema)


def finalize_writer(temp_path: str, final_path: str) -> None:
    if os.path.exists(final_path):
        os.remove(final_path)
    os.replace(temp_path, final_path)
