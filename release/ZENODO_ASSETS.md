# Zenodo Assets

This file is the manifest for the Zenodo deposit accompanying the paper.
Heavy artifacts (model weights, interpretability parquets, code archive) are staged under `release/zenodo/` and intentionally ignored by git because they exceed normal GitHub file limits.
Small artifacts (this manifest, the dataset card, dataset metadata, fold definitions, eval JSONs) live under `release/` and are git-trackable.

## Upload manifest

Files to include in the Zenodo deposit, with their staging location:

| Layer | Files | Source |
|---|---|---|
| Landing page | `README.md` | `release/zenodo/README.md` |
| Source code | `code-v1.0.0.tar.gz` | `release/zenodo/code-v1.0.0.tar.gz` |
| AttriMIL Flash weights | `attrimil_p{16s8,8s4}_flash_fold{0-4}.pth` (10 files) | `release/zenodo/` |
| Baseline weights | `{convnextv2_base,swin_s3_base_224}_{book,page}_split_fold{0-4}.pth` (20 files) | `release/zenodo/` |
| Interpretability | `attrimil_8x8_patch_embeddings_clusters.parquet`, `cluster_assignments_8x8.parquet` | `release/zenodo/` |
| Dataset metadata | `dataset_metadata.parquet`, `dataset_metadata.csv.gz` | `release/` |
| Fold definitions | `folds_{book,page}_5fold.json`, `folds_{book,page}_5fold_stats.json` | `release/` |
| Eval results | `attrimil_p{16s8,8s4}_flash_v2_eval.json` | `release/` |
| Baseline eval results | 20 JSON files | `release/baseline_eval/` |
| Documentation | `DATASET_CARD.md`, `ZENODO_ASSETS.md` (this file) | `release/` |

Total: ~64 files, ~9.1 GB.

## Licensing

The Zenodo record is published under **CC-BY-4.0**, which covers the dataset metadata, fold definitions, model weights, interpretability artifacts, and evaluation results.

The source code archive (`code-v1.0.0.tar.gz`) carries an internal **MIT License** (`LICENSE` file inside the tarball).
The dual-licensing reflects the standard distinction between software (MIT) and data/derived artifacts (CC-BY-4.0).

The page images referenced by `dataset_metadata.parquet` are not part of this record and remain under the Hyundam Mun'go catalog terms.

## Source code archive

| File | Size | SHA-256 |
|---|---:|---|
| `code-v1.0.0.tar.gz` | 13 MB | `29fa67442f5d24b2ca53e3e4dac1a1568175cdf99e2932b74a1b66c1761dac22` |

The archive contains the source tree of `aronvandepol/visual-feature-discovery` at the v1.0.0 release.
Built with `tar` from the working tree at the tagged commit, with virtualenvs, caches, results, and gitignored heavy artifacts excluded.
Prefix dir inside the tarball is `visual-feature-discovery-v1.0.0/`.

## Trained Model Weights

5-fold AttriMIL Flash checkpoints (best per fold by validation loss) for both patch configurations on book-level GroupKFold splits.
Optimizer state has been stripped; each file contains the model state_dict, hyperparameters, and the epoch index.

### 16x16 patches, stride 8

| File | Size | Best epoch | SHA-256 |
|---|---:|---:|---|
| `attrimil_p16s8_flash_fold0.pth` | 12,690,947 bytes | 20 | `4a042d7f698f629daa5d4f31a27fcc37bfd3697cb85de3497547189e6264295a` |
| `attrimil_p16s8_flash_fold1.pth` | 12,690,947 bytes | 17 | `5daed4ed9e146be2690c7d1f52cc51aba8341f80778ef21d3c7784d9be4ac7cc` |
| `attrimil_p16s8_flash_fold2.pth` | 12,690,947 bytes | 32 | `1a719c95516ae4b3f75f481046ae9c29053b219068912b8416e041cfc99cbe9d` |
| `attrimil_p16s8_flash_fold3.pth` | 12,690,947 bytes | 32 | `94cc31c9f8bdd39756b42d7a4a91c07c54c6d0b9ba1ea0101f1c0d568847d499` |
| `attrimil_p16s8_flash_fold4.pth` | 12,690,947 bytes | 14 | `ca7ec40375852a3bf9e59e21ddcf6c50f6618f436f2d82d41b709f6e2b0b033e` |

5-fold mean test metrics (recorded in `release/attrimil_p16s8_flash_v2_eval.json`):

| Metric | Mean | Std (sample) |
|---|---:|---:|
| Accuracy | 84.52% | 2.19% |
| F1 | 0.834 | 0.019 |
| Precision | 0.844 | 0.019 |
| Recall | 0.830 | 0.024 |

### 8x8 patches, stride 4

| File | Size | Best epoch | SHA-256 |
|---|---:|---:|---|
| `attrimil_p8s4_flash_fold0.pth` | 12,690,827 bytes | 14 | `91bab030e541f44f0ef99243a4a3c49220d0c7ef28e67ffbdda20db6548d6ca8` |
| `attrimil_p8s4_flash_fold1.pth` | 12,690,827 bytes | 25 | `7af8f6cb8f68774166e20ffbbfe7c892d2756c216ba413765278426c8159d0af` |
| `attrimil_p8s4_flash_fold2.pth` | 12,690,827 bytes | 20 | `57da14c9529e5866ce2c951ceb39f8cc062d390de34a2a6af788a1e618464f26` |
| `attrimil_p8s4_flash_fold3.pth` | 12,690,827 bytes | 62 | `965c0b7611c7e171c4e251acbf008a0ca7776dd816c38d289052da59741390be` |
| `attrimil_p8s4_flash_fold4.pth` | 12,690,827 bytes | 24 | `3e4012b4e68683cb102fee0dd7d5ce22e07620104fd4955c835eff0a0d5fae99` |

5-fold mean test metrics (recorded in `release/attrimil_p8s4_flash_v2_eval.json`):

| Metric | Mean | Std (sample) |
|---|---:|---:|
| Accuracy | 83.60% | 3.24% |
| F1 | 0.825 | 0.037 |
| Precision | 0.828 | 0.034 |
| Recall | 0.823 | 0.040 |

Total: 10 weight files, ~120 MB.

## Baseline Model Weights

5-fold ConvNeXtV2 Base and Swin S3 Base 224 baseline checkpoints (best per fold by validation loss) under both evaluation protocols.
Optimizer state has been stripped; each file contains the model state_dict, hyperparameters, epoch index, and `pytorch-lightning_version`.
Loading them requires the matching Lightning module from the training scripts (`train_book_split.py` and `train_page_split.py`).
Per-fold eval JSONs are bundled separately under `release/baseline_eval/`.

### ConvNeXtV2 Base (full page, 896x896)

#### Held-out-book (GroupKFold)

| File | Size | Best epoch | SHA-256 |
|---|---:|---:|---|
| `convnextv2_base_book_split_fold0.pth` | 350,942,393 bytes | 5 | `66d9ed26a754ebecf30e497400bada0fcd4b453be8572f9b0af4cf0d5437a295` |
| `convnextv2_base_book_split_fold1.pth` | 350,942,393 bytes | 7 | `87dadccafcaf26c2b55e0d66890a6a93ad284b63df1bccb60d8788419ecde23c` |
| `convnextv2_base_book_split_fold2.pth` | 350,942,393 bytes | 0 | `c268dcb21ea72947d1e7cd9162a58a3da90a032af6cf99e400fa2d0de67b0fe7` |
| `convnextv2_base_book_split_fold3.pth` | 350,942,393 bytes | 1 | `f77f8f74439f3361b93187d235a2a52d9ffcba12da488c7047c40920187c2b92` |
| `convnextv2_base_book_split_fold4.pth` | 350,942,393 bytes | 15 | `b1e7754df92c3cdc165a3fe6215836775f352d6a358a830800e9f51b3612108c` |

5-fold mean: 81.98 +/- 2.40% accuracy.

#### Page-random (KFold)

| File | Size | Best epoch | SHA-256 |
|---|---:|---:|---|
| `convnextv2_base_page_split_fold0.pth` | 350,942,393 bytes | 17 | `fd9aeca9a3c336dff46aadf600c0e7cab0ccb37e8f1caf5dfa7e87c01fd4035c` |
| `convnextv2_base_page_split_fold1.pth` | 350,942,393 bytes | 9 | `5d4b4417728cf8178b04f2d93266a135cfada9e21c342c6715c22d90fe6c2f41` |
| `convnextv2_base_page_split_fold2.pth` | 350,942,393 bytes | 36 | `c242078ef7707d22326a314f090c6f05478d88ac80babaf971292af19cf8290e` |
| `convnextv2_base_page_split_fold3.pth` | 350,942,393 bytes | 10 | `51f4bddd5a2d112560948ec38ed3cf41d0bf46fd97425de11079910f26008893` |
| `convnextv2_base_page_split_fold4.pth` | 350,942,393 bytes | 30 | `e7c54c8ef011f042e8ff4ef72d0c1c42ac1a302b96592802cbd4dc048fc01e24` |

5-fold mean: 98.05 +/- 0.24% accuracy.

### Swin S3 Base 224 (page cutouts)

#### Held-out-book (GroupKFold)

| File | Size | Best epoch | SHA-256 |
|---|---:|---:|---|
| `swin_s3_base_224_book_split_fold0.pth` | 281,644,307 bytes | 0 | `66011c059c75d8d69f64c71b8004ae23d5506d4b5a173cbcbb8ecee1c013aa0b` |
| `swin_s3_base_224_book_split_fold1.pth` | 281,644,307 bytes | 12 | `333de09621eb053a0d2daf7d09719fa585ebea365f3950347593815f567506a2` |
| `swin_s3_base_224_book_split_fold2.pth` | 281,644,307 bytes | 6 | `135b20d07df12dd2cea97615f5c1f0d4e90ee95870cc4befd3852105cc7b9462` |
| `swin_s3_base_224_book_split_fold3.pth` | 281,644,307 bytes | 1 | `c1cbd7e42249514fff5aa3e0e3875abef427d08c461da48de654a74f29db59b7` |
| `swin_s3_base_224_book_split_fold4.pth` | 281,644,307 bytes | 17 | `6fb70387dc608ff6c474754e9a51bf472f664c9ae88c93296870e907b09e56fb` |

5-fold mean: 84.11 +/- 1.81% accuracy.

#### Page-random (KFold)

| File | Size | Best epoch | SHA-256 |
|---|---:|---:|---|
| `swin_s3_base_224_page_split_fold0.pth` | 281,644,307 bytes | 81 | `622f2bd530d17d039ad2bfe8c750fccc6296f60ae864930bd600e941d8dcff06` |
| `swin_s3_base_224_page_split_fold1.pth` | 281,644,307 bytes | 70 | `368991a9f6a847244715285bacdfca2b23e06a8f3f30bf0b9564a6772ec939dc` |
| `swin_s3_base_224_page_split_fold2.pth` | 281,644,307 bytes | 70 | `cf53f5537e459caa02c8773bfc2ca728879525c8a1c86a1088efd422412840de` |
| `swin_s3_base_224_page_split_fold3.pth` | 281,644,307 bytes | 67 | `f38814406f5c982f0a1ff4b77feaec3923c7fe4fe8b469178e6d5a29baa6fe07` |
| `swin_s3_base_224_page_split_fold4.pth` | 281,644,307 bytes | 55 | `0809dbd6a76fe08685dd3adf584a45a4de52b5fe4668df27cc23bd797d7391a4` |

5-fold mean: 98.15 +/- 0.17% accuracy.

Total baselines: 20 weight files, ~6.3 GB combined.
Verified means agree with the paper's `table:comparative_performance`: ConvNeXt 82.0% / 98.1%, Swin 84.1% / 98.2%.

## Interpretability Artifacts

| File | Size | Rows | Description | SHA-256 |
|---|---:|---:|---|---|
| `attrimil_8x8_patch_embeddings_clusters.parquet` | 3,130,193,698 bytes | 3,627,572 | Full 8x8 high-attention patch artifact with embedding vectors, attention weights, UMAP/t-SNE coordinates, HDBSCAN cluster labels, and probabilities. | `1050bffc0b38a6f9514d0831541c3ee440c74b962daebd21ba69148a9b8a7e82` |
| `cluster_assignments_8x8.parquet` | 124,770,945 bytes | 3,627,572 | Compact 8x8 cluster-assignment artifact without embedding vectors. Includes normalized image paths, canonical printshop labels, patch coordinates, attention weights, UMAP/t-SNE coordinates, cluster labels, and probabilities. | `e725a67ce27498fdbccb6a64da3fcfa50476eee13d81cfdacb112dce56e4144b` |

## Notes

The full interpretability artifact corresponds to the 8x8 Multiple Instance Learning model used for the paper's interpretability analysis.
The compact artifact is derived from the full artifact and omits the `embedding` column.
Neither interpretability artifact contains page-image data.

The 10 AttriMIL weight files use the AttriMIL Flash architecture (multi-head Q/K/V attention with output projection).
Loading them requires `LightningAttriMILFlash` from `src.models.lightning_attrimil_flash`, with `patch_size` and `stride` matching the filename (16/8 or 8/4) and `num_classes=4`.

The 20 baseline weight files come from the training scripts in the project root (`train_book_split.py` for held-out-book GroupKFold and `train_page_split.py` for page-random KFold).
ConvNeXtV2 Base operates on full pages at 896x896; Swin S3 Base 224 operates on text-only page cutouts.
All baselines use `num_classes=4` over the canonicalized printshop set (Adan Mungo merge for Taedong applied).
