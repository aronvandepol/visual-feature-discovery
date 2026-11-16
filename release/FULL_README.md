# Visual Feature Discovery for Colonial Korean Print — Release Bundle

This Zenodo record is the reproducibility bundle for the article *Interpretable Visual Feature Discovery using Multiple Instance Learning: A Case Study of Colonial Korean Print*, published in *Computational Humanities Research* 2, e22 (Cambridge University Press, 2026), https://doi.org/10.1017/chr.2026.10036.

It contains the source code, trained model weights, dataset metadata, fold definitions, and interpretability artifacts required to reproduce the paper's training, evaluation, and clustering workflows.

## Source code

The training, evaluation, and interpretability code is bundled as `code-v1.0.0.tar.gz` and contains the source tree of `aronvandepol/visual-feature-discovery` at version 1.0.0 (commit `dedcfe3`).
The active development repository is at:

`https://github.com/aronvandepol/visual-feature-discovery`

Issues and questions about reproduction should be filed there.

## Trained model weights

All weights are released with optimizer state stripped, so each `.pth` file holds the model `state_dict`, hyperparameters, epoch index, and `pytorch-lightning_version`.

### Multiple Instance Learning models

5-fold AttriMIL Flash checkpoints for the two patch configurations evaluated in the paper.

- `attrimil_p16s8_flash_fold{0-4}.pth` — 16x16 patches with stride 8, ~12 MB each.
- `attrimil_p8s4_flash_fold{0-4}.pth` — 8x8 patches with stride 4, ~12 MB each.

Loading requires `LightningAttriMIL` from `src.models.lightning_attrimil` in the source archive.
Each file stores its own `hyper_parameters` (`num_classes=4`, `patch_size`, `stride`, class names), so `LightningAttriMIL(**ckpt["hyper_parameters"])` followed by `load_state_dict(ckpt["state_dict"])` restores the model.

### Baselines

5-fold ConvNeXtV2 Base and Swin S3 Base 224 checkpoints under both evaluation protocols (held-out-book GroupKFold and page-random KFold), 20 files in total.

- `convnextv2_base_{book,page}_split_fold{0-4}.pth` — ~351 MB each.
- `swin_s3_base_224_{book,page}_split_fold{0-4}.pth` — ~282 MB each.

Loading requires `LightningTIMM` from `src.models.lightning_timm` in the source archive, with `num_classes=4` and the `model_name` stored in each file's `hyper_parameters` (`convnextv2_base` or `swin_s3_base_224`).
The training scripts are `scripts/training/train_baseline_book.py` and `scripts/training/train_baseline_page.py`.

## Interpretability artifacts

8x8 high-attention patch outputs from the AttriMIL Flash model used for the paper's clustering analysis.

- `attrimil_8x8_patch_embeddings_clusters.parquet` — full artifact with embedding vectors, attention weights, UMAP and t-SNE coordinates, and HDBSCAN cluster assignments. ~3.1 GB.
- `cluster_assignments_8x8.parquet` — compact variant without embedding vectors. ~119 MB.

## Dataset metadata and fold definitions

- `dataset_metadata.parquet` and `dataset_metadata.csv.gz` — page-level metadata for the 57,583 pages used in the paper, with canonical printshop labels and source catalog links.
- `folds_book_5fold.json` and `folds_page_5fold.json` — fold definitions for the held-out-book and page-random protocols.
- `folds_book_5fold_stats.json` and `folds_page_5fold_stats.json` — fold summary statistics.

## Evaluation results

Per-fold eval JSONs for every released model, used to verify that the bundled weights reproduce the paper's headline numbers.

- `attrimil_p16s8_flash_v2_eval.json`, `attrimil_p8s4_flash_v2_eval.json` — AttriMIL Flash 5-fold results.
- 20 baseline eval JSONs, one per baseline weight file and with the same basename (for example `convnextv2_base_book_split_fold0.json`).

## Documentation

- `DATASET_CARD.md` — dataset description, source collection, metadata fields, intended use.
- `ZENODO_ASSETS.md` — full manifest with file sizes, best epochs, SHA-256 checksums, and 5-fold metrics tables.

## Page-image corpus

The page-image corpus is not part of this record.
Ten sample pages for testing the pipeline are the only images included (`sample_data/` in the code archive).
Images are accessible via the Hyundam Mun'go digital catalog, with per-page URLs provided in `dataset_metadata.parquet`.
The collection was previously known as Adan Mungo.

## License

The deposit is released under **CC-BY-4.0** (Creative Commons Attribution 4.0 International), which applies to the dataset metadata, fold definitions, model weights, interpretability artifacts, and evaluation results.

The source code archive (`code-v1.0.0.tar.gz`) is internally licensed under the **MIT License**, included as the `LICENSE` file inside the tarball.
This dual-licensing follows standard practice for mixed software-and-data deposits.

The page-image corpus is not part of this record and remains governed by the Hyundam Mun'go catalog terms.

## Citation

Cite this Zenodo record using the **concept DOI**, which always resolves to the latest version:

`10.5281/zenodo.20015055`

This is version 2.0.0 of the record, the full release.
Version 1.0.0 (`10.5281/zenodo.20015056`) was a pre-publication placeholder that held only a README.

Cite the article as:

van de Pol, A., Prokic, J., & Mol, A. (2026). Interpretable visual feature discovery using multiple instance learning: A case study of colonial Korean print. *Computational Humanities Research*, 2, e22. https://doi.org/10.1017/chr.2026.10036

## Contact

Aron van de Pol, Leiden University.
Issues and questions about reproduction should be filed at the GitHub repository linked above.
