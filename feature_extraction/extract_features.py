#!/usr/bin/env python3
"""Minimal Comet integration helpers restored from the original feature extractor."""

from __future__ import annotations

import ast
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

try:
    from comet_ml import API
except ImportError:
    API = None

from feature_extraction.config import (
    COMET_WORKSPACE,
    MODEL_CACHE_DIR,
    PRINTSHOP_CANONICAL_MAP,
    USE_COMET_MODELS,
)
from src.models.lightning_attrimil import LightningAttriMIL  # noqa: F401 (type ref)


def _load_comet_key_from_constants() -> Optional[str]:
    """Read the Comet API key from an override file when env var is absent.

    Set COMET_API_KEY environment variable or point COMET_API_KEY_FILE to a
    constants.json file with {"api_keys": {"comet": "YOUR_KEY"}}.
    """
    override = os.environ.get("COMET_API_KEY_FILE")
    if override:
        candidates = [Path(override).expanduser()]
    else:
        candidates = [
            Path.cwd() / "constants.json",
            Path(__file__).resolve().parent.parent / "constants.json",
        ]

    for path in candidates:
        if not path or not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        api_key = data.get("api_keys", {}).get("comet")
        if api_key:
            return str(api_key)
    return None


def _normalize_class_name_list(raw: Optional[Any]) -> Optional[list[str]]:
    """Best-effort parse of class name lists stored as strings or sequences."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return None
        raw = parsed
    if isinstance(raw, (list, tuple)):
        names: list[str] = []
        for item in raw:
            if item is None:
                continue
            names.append(str(item))
        return names or None
    return None


def _canonicalize_label(name: str) -> str:
    return PRINTSHOP_CANONICAL_MAP.get(name, name)


def _align_classifier_head(
    state_dict: Dict[str, torch.Tensor],
    checkpoint_class_names: Optional[list[str]],
    target_class_names: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Align a checkpoint classifier head with the desired target classes.

    When the checkpoint was trained with more classes (e.g., pre-canonicalisation),
    collapse multiple source classes that map to the same canonical label by averaging
    their weights and biases. Falls back to a naive slice when metadata is missing.
    """
    weight_key = "model.classifier.weight"
    bias_key = "model.classifier.bias"
    weight = state_dict.get(weight_key)
    bias = state_dict.get(bias_key)
    target_class_names = list(target_class_names)

    if weight is None or bias is None:
        return state_dict

    target_count = len(target_class_names)
    if target_count == 0 or weight.shape[0] == target_count:
        return state_dict

    source_count = weight.shape[0]
    if source_count < target_count:
        # Cannot expand classifier; leave as-is and let load_state_dict raise.
        return state_dict

    canonical_source_names: list[str] = []
    if checkpoint_class_names and len(checkpoint_class_names) == source_count:
        canonical_source_names = [_canonicalize_label(n) for n in checkpoint_class_names]
    else:
        canonical_source_names = []

    new_weight = weight.new_zeros((target_count, weight.shape[1]))
    new_bias = bias.new_zeros((target_count,))

    for idx, target_name in enumerate(target_class_names):
        target_canonical = _canonicalize_label(target_name)
        matches: list[int] = []
        if canonical_source_names:
            matches = [
                i for i, src_name in enumerate(canonical_source_names)
                if src_name == target_canonical
            ]
        if not matches:
            # Fallback: take the corresponding row if within bounds, else last row.
            fallback_index = min(idx, source_count - 1)
            matches = [fallback_index]
        match_weights = weight[matches, :]
        match_bias = bias[matches]
        new_weight[idx] = match_weights.mean(dim=0)
        new_bias[idx] = match_bias.mean()

    state_dict[weight_key] = new_weight
    state_dict[bias_key] = new_bias
    return state_dict


def setup_comet_api():
    """Instantiate a Comet API client, ensuring caching directory exists."""
    if API is None:
        raise ImportError("comet_ml is required for model downloading. Install with: pip install comet_ml")
    if not USE_COMET_MODELS:
        raise RuntimeError("COMET integration disabled (USE_COMET_MODELS=False)")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        api_key = _load_comet_key_from_constants()
    if not api_key:
        raise ValueError("COMET_API_KEY is required. Set the environment variable or provide constants.json.")
    os.environ.setdefault("COMET_API_KEY", api_key)
    return API(api_key=api_key)


def find_and_load_model_from_comet(api, project_name, experiment_name, fold, num_classes, class_names):
    """
    Find, download and load model from Comet ML experiment
    Enhanced with debugging and error handling like evaluate_model.py

    Returns:
        tuple: (loaded_model, experiment_metadata, temp_dir)
    """

    print(f"Loading model for {experiment_name}_fold{fold} from Comet ML...")
    print(f"Searching for experiments in project: {project_name}")

    # Debug: Check available workspaces and projects
    try:
        workspace = api.get_default_workspace() if COMET_WORKSPACE is None else COMET_WORKSPACE
        print(f"Using workspace: {workspace}")

        # List all projects in workspace to help debug
        try:
            all_projects = api.get_projects(workspace=workspace)
            project_names = list(all_projects)
            print(f"Available projects in workspace '{workspace}': {project_names}")

            # Check for similar project names
            matching_projects = [p for p in project_names if project_name.lower() in p.lower() or p.lower() in project_name.lower()]
            if matching_projects:
                print(f"Similar project names found: {matching_projects}")

            # Check exact match
            if project_name in project_names:
                print(f"✅ Exact project match found: {project_name}")
            else:
                print(f"❌ Project '{project_name}' not found in available projects")

        except Exception as e:  # pragma: no cover - diagnostics only
            print(f"Could not list projects: {e}")

    except Exception as e:
        print(f"Could not get default workspace: {e}")
        workspace = COMET_WORKSPACE

    # Get all experiments from the project
    experiments = api.get_experiments(workspace=workspace, project_name=project_name)

    # Debug: List all experiment names to see the actual pattern
    experiment_names = []
    for exp in experiments:
        exp_name = exp.name
        if exp_name:
            experiment_names.append(exp_name)

    print(f"All experiments in project: {experiment_names}")

    # Find matching experiment
    candidate_names: list[str] = []
    suffix = f"_fold{fold}"
    def _add_candidate(name: Optional[str]) -> None:
        if not name:
            return
        if name not in candidate_names:
            candidate_names.append(name)

    _add_candidate(experiment_name)
    if not experiment_name.endswith(suffix):
        _add_candidate(f"{experiment_name}{suffix}")
    else:
        base_no_suffix = experiment_name[: -len(suffix)]
        _add_candidate(base_no_suffix)
        _add_candidate(f"{base_no_suffix}_{fold}")
        _add_candidate(f"{base_no_suffix}_{fold}{suffix}")
    _add_candidate(f"{experiment_name}_{fold}")

    print(f"Looking for experiment matching one of: {candidate_names}")

    matching_experiments = []
    for exp in experiments:
        exp_name = exp.name
        if exp_name and exp_name in candidate_names:
            matching_experiments.append(exp)

    if not matching_experiments:
        print("No direct match found; trying broader pattern search")
        alt_patterns = [
            f"{experiment_name}_fold{fold}",
            f"{experiment_name}_{fold}",
            f"{experiment_name}{suffix}",
        ]
        for pattern in alt_patterns:
            if pattern in candidate_names:
                continue
            print(f"Trying alternative pattern: {pattern}")
            for exp in experiments:
                exp_name = exp.name
                if exp_name and pattern == exp_name:
                    matching_experiments.append(exp)
                    break
            if matching_experiments:
                break

    if not matching_experiments:
        raise ValueError(
            f"No experiments found matching any of {candidate_names} in project '{project_name}'. "
            f"Available experiments: {experiment_names}"
        )

    # Use the first matching experiment
    matching_experiment = matching_experiments[0]
    exp_key = matching_experiment.key
    exp_name = matching_experiment.name

    print(f"Found experiment: {exp_name} (key: {exp_key})")

    # Get experiment details and download model
    exp_details = api.get_experiment(workspace=workspace, project_name=project_name, experiment=exp_key)

    # Download model from assets
    print("Looking for model files in experiment assets...")
    assets = exp_details.get_asset_list()

    if not assets:
        raise ValueError(f"No assets found in experiment {exp_name}")

    # Find model file - prioritize final_model.pth like evaluate_model.py
    print(f"Found {len(assets)} assets")

    # Look for model files (typically .pth files)
    model_assets = []
    for asset in assets:
        asset_name = asset['fileName']
        if asset_name.endswith('.pth') and ('final_model' in asset_name.lower() or 'model' in asset_name.lower()):
            model_assets.append(asset)

    if not model_assets:
        raise ValueError(f"No .pth model files found in experiment {matching_experiment.name} assets")

    print(f"Found {len(model_assets)} model files: {[a['fileName'] for a in model_assets]}")

    # Prefer 'final_model.pth' if available, otherwise use first model file
    model_asset = None
    for asset in model_assets:
        if asset['fileName'] == 'final_model.pth':
            model_asset = asset
            break

    if not model_asset:
        model_asset = model_assets[0]  # Use first available model file

    print(f"Downloading model: {model_asset['fileName']}")

    # Download model
    temp_dir = tempfile.mkdtemp(prefix=f"extract_features_{matching_experiment.key}_", dir=MODEL_CACHE_DIR)
    model_path = os.path.join(temp_dir, model_asset['fileName'])

    asset_response = exp_details.get_asset(
        asset_id=model_asset['assetId'],
        return_type="response",
        stream=True
    )

    # Download the model asset using streaming
    print(f"Downloading to: {model_path}")

    with open(model_path, 'wb') as fd:
        for chunk in asset_response.iter_content(chunk_size=1024 * 1024):
            fd.write(chunk)

    print(f"Successfully downloaded model: {model_asset['fileName']} ({os.path.getsize(model_path)} bytes)")

    # Get experiment parameters
    param_list = exp_details.get_parameters_summary()
    params = {p['name']: p['valueCurrent'] for p in param_list}

    # Load model
    try:
        checkpoint_obj = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = checkpoint_obj.get('state_dict', checkpoint_obj)
        ckpt_class_names = _normalize_class_name_list(
            checkpoint_obj.get('hyper_parameters', {}).get('class_names')
            if isinstance(checkpoint_obj, dict) else None
        )
        if not ckpt_class_names:
            ckpt_class_names = _normalize_class_name_list(params.get('class_names'))

        # Extract hyperparameters
        learning_rate = float(params.get('learning_rate', 1e-4))
        alpha = float(params.get('alpha', 0.1))
        beta = float(params.get('beta', 0.01))
        patch_size = int(params.get('patch_size', 8))
        stride = int(params.get('stride', 4))

        print(f"Model parameters: patch_size={patch_size}, stride={stride}")
        if ckpt_class_names:
            print(f"Checkpoint classes: {ckpt_class_names}")

        model = LightningAttriMIL(
            num_classes=num_classes,
            class_names=class_names,
            learning_rate=learning_rate,
            alpha=alpha,
            beta=beta,
            patch_size=patch_size,
            stride=stride
        )

        state_dict = _align_classifier_head(state_dict, ckpt_class_names, class_names)
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys, unexpected_keys = load_result
        if missing_keys:
            print(f"⚠️ Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys when loading checkpoint: {unexpected_keys}")

        model.eval()

        # Validate model parameters match expectations
        print("Model loaded successfully:")
        print(f"  - Patch size: {patch_size}")
        print(f"  - Stride: {stride}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Alpha: {alpha}, Beta: {beta}")

        metadata = {
            'experiment_name': exp_name,
            'experiment_key': exp_key,
            'patch_size': patch_size,
            'stride': stride,
            'parameters': params,
            'model_file': model_asset['fileName']
        }

        return model, metadata, temp_dir

    except Exception as e:
        shutil.rmtree(temp_dir)
        raise ValueError(f"Failed to load model: {e}")
