# Sample Data

A small subset (10 pages) of real historical printshop page images for testing
and verifying your installation. These are actual scanned document pages, not
synthetic data.

## Directory structure

```
sample_data/
├── README.md                        # This file
├── images/
│   ├── 대동인쇄소/                   # 3 pages
│   ├── 한성도서 주식회사/              # 3 pages
│   ├── 신문관/                       # 2 pages
│   └── 조선인쇄 주식회사/              # 2 pages
├── sample_dataset.parquet           # 10-row dataset
└── sample_folds.json                # 2-fold book-level splits
```

## Parquet schema (`sample_dataset.parquet`)

| Column       | Type | Description |
|-------------|------|-------------|
| `filename`   | str  | Original filename, same as basename of `image_path` |
| `image_path` | str  | Relative path inside `images/`, e.g. `대동인쇄소/주보 신생활 15호-4.jpg` |
| `printshop`  | str  | Printshop (class) name as it appears in the source data |
| `book`       | str  | Book title — all pages from the same physical book share this value |
| `book_id`    | str  | Unique book identifier (`{printshop}_{book}`) |

## Printshop canonicalization

The data module canonicalizes variant printshop names automatically.
The default mapping merges two Daedong (대동) variants:

| Original name | Canonical name |
|--------------|----------------|
| `대동인쇄 주식회사 大東印刷 株式會社` | `대동인쇄소` |
| `대동인쇄소 大東印刷所`              | `대동인쇄소` |

Other printshop names (`한성도서 주식회사`, `신문관`, `조선인쇄 주식회사`) pass through unchanged.

## Fold file (`sample_folds.json`)

Two folds using book-level GroupKFold splits (no book appears in both train
and validation within the same fold):

- **Fold 0**: train = 신문관, 조선인쇄 주식회사 (4 pages) / val = 대동인쇄소, 한성도서 주식회사 (6 pages)
- **Fold 1**: train = 대동인쇄소, 한성도서 주식회사 (6 pages) / val = 신문관, 조선인쇄 주식회사 (4 pages)

Format matches `generate_folds_book_split.py` output.

## Quick-start usage

```bash
# Verify the parquet file
python -c "import pandas as pd; print(pd.read_parquet('sample_data/sample_dataset.parquet'))"

# Verify the fold file
python -c "import json; print(json.load(open('sample_data/sample_folds.json'))['metadata'])"

# Verify images load correctly
python -c "from PIL import Image; img = Image.open('sample_data/images/대동인쇄소/주보 신생활 15호-4.jpg'); print(f'{img.size}, mode={img.mode}')"
```
