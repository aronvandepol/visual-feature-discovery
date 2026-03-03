#!/usr/bin/env python3
"""Headless patch extraction library.

Processes pages one-by-one, computes per-page normalized attention, selects
patches above a percentile threshold, and emits rows compatible with
feature_extraction.schemas.get_core_schema().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from feature_extraction.config import (
    EXPERIMENT_NAME,
    DEFAULT_FOLD,
    COMET_PROJECT_NAME,
    get_image_path_resolver,
    get_class_names,
)
from feature_extraction.helpers import (
    preprocess_image,
    tensor_to_patch_grid,
    normalize_minmax,
    select_by_percentile,
)

# Reuse robust Comet helpers
from feature_extraction.extract_features import setup_comet_api, find_and_load_model_from_comet
from src.models.lightning_attrimil import LightningAttriMIL  # noqa: F401 (type ref)


@dataclass
class ExtractorConfig:
    percentile: float = 99.5
    input_size: int = 896
    patch_size: int = 16
    stride: int = 8
    device: Optional[str] = None
    save_embeddings: bool = False
    experiment_name: Optional[str] = None
    fold: Optional[int] = None
    project_name: Optional[str] = None


class ConfigDrivenAttentionExtractor:
    def __init__(self, cfg: ExtractorConfig):
        self.cfg = cfg
        if cfg.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        self._model = None
        self._resolver = get_image_path_resolver()
        self.class_names = None
        self._experiment_name = self.cfg.experiment_name or EXPERIMENT_NAME
        self._fold = DEFAULT_FOLD if self.cfg.fold is None else int(self.cfg.fold)
        self._project_name = self.cfg.project_name or COMET_PROJECT_NAME

    @property
    def model(self):
        if self._model is None:
            api = setup_comet_api()
            class_names = list(get_class_names())
            project_name = self._project_name
            experiment_name = self._experiment_name
            fold = self._fold
            model, metadata, _tmp = find_and_load_model_from_comet(
                api,
                project_name=project_name,
                experiment_name=experiment_name,
                fold=fold,
                num_classes=len(class_names),
                class_names=class_names,
            )
            # Global speed knobs
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.set_float32_matmul_precision('high')  # enable TF32 on Ampere
            except Exception:
                pass

            m = model.to(self.device).eval()
            # Try Inductor compile for extra speed; safe fallback if unsupported
            try:
                m = torch.compile(m, mode="max-autotune", fullgraph=False)
            except Exception:
                pass
            self._model = m
            self.class_names = class_names
        return self._model

    def process_row(self, row: Mapping, page_index: int) -> List[Dict]:
        """Process a single parquet row; return list of output rows for selected patches."""
        image_path_orig = str(row.get("image_path", ""))
        image_path = self._resolver(image_path_orig) if image_path_orig else None
        if not image_path:
            return []

        try:
            img_t = preprocess_image(image_path, input_size=self.cfg.input_size)
        except Exception:
            return []

        # Move preprocessed image to device first; build grid on device to reduce H2D volume
        img_t = img_t.to(self.device, non_blocking=True)
        grid, gh, gw = tensor_to_patch_grid(
            img_t, patch_size=self.cfg.patch_size, stride=self.cfg.stride
        )
        # Reshape flattened patch grid to match AttriMIL's expected input
        grid = grid.reshape(
            1, int(gh), int(gw), grid.shape[1], grid.shape[2], grid.shape[3]
        )

        with torch.no_grad():
            # Prefer bfloat16 autocast on Ampere; fallback to fp16 else keep fp32
            if self.device.type == "cuda":
                use_bf16 = torch.cuda.get_device_capability(self.device)[0] >= 8
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
                try:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        y_prob, y_hat, A_batch, H_batch, _ = self.model(grid)
                except Exception:
                    # Fallback without autocast
                    y_prob, y_hat, A_batch, H_batch, _ = self.model(grid)
            else:
                # Non-CUDA path: run in fp32 (safer on CPU/MPS)
                y_prob, y_hat, A_batch, H_batch, _ = self.model(grid)

        A = A_batch.reshape(-1).float()
        H = H_batch.reshape(-1, H_batch.shape[-1]).float() if H_batch is not None else None
        A_norm = normalize_minmax(A)
        mask, thr = select_by_percentile(A_norm, self.cfg.percentile)

        publication_raw = row.get("publication_date")
        publication_date = str(publication_raw) if publication_raw else None
        publication_year = None
        if publication_date:
            match = re.search(r"(1[89]\d{2}|20\d{2})", publication_date)
            if match:
                try:
                    publication_year = int(match.group(0))
                except ValueError:
                    publication_year = None

        if not mask.any():
            return []

        ps, st = int(self.cfg.patch_size), int(self.cfg.stride)
        selected_rows: List[Dict] = []
        exp_name = self._experiment_name
        fold = self._fold
        for flat_idx, keep in enumerate(mask.tolist()):
            if not keep:
                continue
            i, j = divmod(flat_idx, gw)
            y0, x0 = int(i * st), int(j * st)
            y1, x1 = y0 + ps, x0 + ps

            out = {
                "image_path": image_path_orig,
                "printshop": str(row.get("printshop", "")),
                "book": str(row.get("book", "")),
                "book_id": str(row.get("book_id", "")),
                "page_index": int(page_index),
                "grid_h": int(gh),
                "grid_w": int(gw),
                "flat_idx": int(flat_idx),
                "h_coord": int(i),
                "w_coord": int(j),
                "pixel_start_h": int(y0),
                "pixel_start_w": int(x0),
                "pixel_end_h": int(y1),
                "pixel_end_w": int(x1),
                "patch_size": int(ps),
                "stride_used": int(st),
                "attention_weight": float(A[flat_idx].detach().cpu().item()),
                "attention_weight_norm": float(A_norm[flat_idx]),
                "percentile_used": float(self.cfg.percentile),
                "model_experiment": exp_name,
                "model_fold": int(fold),
                "publication_date": publication_date,
                "publication_year": publication_year,
            }

            if self.cfg.save_embeddings and H is not None:
                # Store fp16 by default to reduce storage/IO; schema expects float16
                emb = H[flat_idx].detach().cpu().numpy().astype(np.float16)
                out["embedding"] = emb.tolist()

            selected_rows.append(out)

        return selected_rows
