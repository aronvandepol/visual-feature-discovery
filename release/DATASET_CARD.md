# Dataset Card: Colonial Korean Printshop Metadata

## Summary

This release documents the 57,583 page-image corpus used for colonial Korean printshop classification in the accompanying paper.
The full page-image corpus is not included in this repository.
The repository releases page-level metadata, fold definitions, sample data for testing, and scripts for reproducing the training and evaluation workflow.

## Source Collection

The page images come from the Hyundam Mun'go digital catalog.
The collection was previously known as Adan Mungo.
At the time of release, catalog and image URLs in the metadata use the `adanmungo.org` domain.

## Released Files

- `dataset_metadata.parquet`: page-level metadata for the 57,583 pages used in the paper.
- `dataset_metadata.csv.gz`: gzipped CSV version of the same metadata.
- `folds_book_5fold.json`: book-level GroupKFold split definitions.
- `folds_page_5fold.json`: page-level KFold split definitions.
- `folds_book_5fold_stats.json`: summary statistics for the book-level folds.
- `folds_page_5fold_stats.json`: summary statistics for the page-level folds.
- `ZENODO_ASSETS.md`: manifest for large Zenodo-only artifacts.

## Zenodo-Only Artifacts

The 8x8 patch embeddings and cluster assignments are too large for normal GitHub tracking.
They are staged for Zenodo and listed in `ZENODO_ASSETS.md`.
The full artifact includes embedding vectors, attention weights, UMAP and t-SNE coordinates, HDBSCAN cluster assignments, and cluster probabilities.
The compact cluster-assignment artifact omits embedding vectors.

## Metadata Fields

The metadata includes image identifiers, canonical printshop labels, raw source labels, book grouping fields, bibliographic fields, and source catalog links.
The canonical printshop label merges the two Taedong registrations used in the source data into the four-class label set used in the paper.
The raw labels are preserved so the merge remains inspectable.

## Intended Use

The release supports reproduction of the paper's training, evaluation, and interpretability workflows.
It is intended for research on historical print, typographic analysis, document image analysis, and interpretable machine learning.
Researchers who want to reproduce the full pipeline must obtain the page images from the source collection.

## Out Of Scope

The metadata should not be treated as a complete catalog of Hyundam Mun'go holdings.
It covers only the four printshops and page images used in the paper.
It does not include the full image corpus.

## Known Limitations

The corpus reflects survival, digitization, and selection patterns in the source collection.
Image quality varies across scans.
The released fold definitions reproduce the evaluation setup, but they do not remove historical sampling bias from the underlying corpus.
The metadata contains source URLs as recorded at release time, and those URLs may change if the collection updates its domain structure.

## License

The deposit uses dual licensing.

The source code archive (`code-v1.0.0.tar.gz`) is released under the **MIT License** (`LICENSE` file inside the tarball).

The dataset metadata, fold definitions, evaluation JSONs, trained model weights, and interpretability artifacts are released under **CC-BY-4.0** (Creative Commons Attribution 4.0 International).
This is the standard separation of software from data and derived numerical artifacts.

The page images themselves are not part of this release and remain governed by the terms of the Hyundam Mun'go collection.
