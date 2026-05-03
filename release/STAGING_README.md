# Visual Feature Discovery for Colonial Korean Print — Release Bundle

This Zenodo record accompanies the article *Interpretable Visual Feature Discovery using Multiple Instance Learning: A Case Study of Colonial Korean Print*, forthcoming in *Computational Humanities Research* (Cambridge University Press, ISSN 2977-8158).

## Status

This is the **pre-publication staging deposit**.
At this stage the record contains only this placeholder README.
The full release bundle (source code, trained model weights, dataset metadata, fold definitions, evaluation results, and interpretability artifacts) will be uploaded to this same record once the article is accepted for publication.
The Zenodo concept DOI of this record will remain stable across versions and is the citable identifier in the paper.

## Source code

The active development repository is:

`https://github.com/aronvandepol/visual-feature-discovery`

A `code-v1.0.0.tar.gz` archive of the source tree at the v1.0.0 release will be added to this record on acceptance.

## What this record will contain at full release

The full deposit, scheduled for upload upon publication, will include:

### Trained Multiple Instance Learning model weights

5-fold AttriMIL Flash checkpoints for two patch configurations.
Optimizer state stripped, weights only.

- `attrimil_p16s8_flash_fold{0-4}.pth` — 16x16 patches with stride 8.
- `attrimil_p8s4_flash_fold{0-4}.pth` — 8x8 patches with stride 4.

### Trained baseline model weights

5-fold ConvNeXtV2 Base and Swin S3 Base 224 checkpoints under both held-out-book GroupKFold and page-random KFold protocols.
Optimizer state stripped.

- `convnextv2_base_{book,page}_split_fold{0-4}.pth`
- `swin_s3_base_224_{book,page}_split_fold{0-4}.pth`

### Interpretability artifacts

8x8 high-attention patch outputs from the AttriMIL Flash model used for the paper's clustering analysis.

- `attrimil_8x8_patch_embeddings_clusters.parquet` — full artifact with embedding vectors, attention weights, UMAP and t-SNE coordinates, and HDBSCAN cluster assignments.
- `cluster_assignments_8x8.parquet` — compact variant without embedding vectors.

### Dataset metadata and fold definitions

- `dataset_metadata.parquet` and `dataset_metadata.csv.gz` — page-level metadata for the 57,583 pages used in the paper, with canonical printshop labels and source catalog links.
- `folds_book_5fold.json` and `folds_page_5fold.json` — fold definitions for the held-out-book and page-random protocols.
- `folds_book_5fold_stats.json` and `folds_page_5fold_stats.json` — fold summary statistics.

### Evaluation results

Per-fold eval JSONs for every released model.

### Documentation

- `DATASET_CARD.md` — dataset description, source collection, metadata fields, intended use.
- `ZENODO_ASSETS.md` — full manifest with file sizes, best epochs, SHA-256 checksums, and 5-fold metrics tables.

## Page-image corpus

The page-image files themselves are not part of this record.
Images are accessible via the Hyundam Mun'go digital catalog, with per-page URLs provided in `dataset_metadata.parquet` once the metadata is uploaded.
The collection was previously known as Adan Mungo.

## License

The deposit will use dual licensing on full release.

The source code archive (`code-v1.0.0.tar.gz`) will be released under the **MIT License** (`LICENSE` file included inside the tarball).

The dataset metadata, fold definitions, evaluation JSONs, trained model weights, and interpretability artifacts will be released under **CC-BY-4.0** (Creative Commons Attribution 4.0 International).

The page images themselves remain governed by the Hyundam Mun'go catalog terms.

## Citation

Cite this Zenodo record using the **concept DOI**, which always resolves to the latest version:

`10.5281/zenodo.20015055`

The current pre-publication version is v1.0.0 (`10.5281/zenodo.20015056`).
The article and Zenodo record will receive their final citation block on acceptance.

## Contact

Aron van de Pol, Leiden University.
Issues and questions about reproduction should be filed at the GitHub repository linked above.
