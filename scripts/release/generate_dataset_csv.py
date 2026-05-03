# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas>=2.0",
#   "pyarrow>=18",
# ]
# ///
"""
Generate the release-grade dataset metadata CSV and parquet.

Joins the page-level parquet against a local source SQLite metadata database to produce
a single file with the canonical printshop class plus the raw printshop
attribution, and full bibliographic metadata from the source catalog.

Inputs:
  --parquet     Path to the page-level parquet (printshop_dataset_cho.parquet)
  --db          Path to the local source SQLite metadata database
  --out-csv     Output CSV path (default: release/dataset_metadata.csv)
  --out-parquet Output parquet path (default: release/dataset_metadata.parquet)

The output schema is documented at the bottom of this file.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# Canonicalization rules. The two Taedong corporate registrations are merged
# into a single class because they share matrices and personnel. The CSV
# preserves the raw attribution under printshop_raw_korean / printshop_raw_hanja
# so the merge is transparent and reversible.
TAEDONG_CANONICAL_KOREAN = "대동인쇄소"
TAEDONG_CANONICAL_HANJA = "大東印刷所"
PRINTSHOP_CANON_MAP = {
    # raw_korean -> canonical_korean
    "대동인쇄 주식회사": TAEDONG_CANONICAL_KOREAN,
    "대동인쇄소": TAEDONG_CANONICAL_KOREAN,
}


def nfc(s):
    if isinstance(s, str):
        return unicodedata.normalize("NFC", s)
    return s


def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = {"printshop", "book", "book_id", "filename", "image_path", "original_image_path"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"Parquet missing required columns: {missing}")
    # Drop the schema-only columns; we will repopulate from the DB join.
    keep = list(needed)
    df = df[keep].copy()
    # Normalize all text columns to NFC.
    for col in keep:
        df[col] = df[col].apply(nfc)
    # Compute the book directory (used as join key against the DB).
    df["book_dir"] = df["original_image_path"].apply(
        lambda p: nfc(os.path.dirname(p)) if isinstance(p, str) else None
    )
    return df


def load_db(path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(path))
    db = pd.read_sql_query(
        """
        SELECT
            m.File_Location,
            m.Title_Korean   AS title_korean,
            m.Title_Hanja    AS title_hanja,
            m.Title_Other    AS title_other,
            m.Issue_Number   AS issue_number,
            m.Total_Issue_Number AS total_issue_number,
            m.Frequency      AS frequency,
            m.Category       AS category,
            m.Publication_Date AS publication_date,
            m.Printing_Date  AS printing_date,
            m.Page_Count     AS page_count_in_volume,
            m.Width          AS width_mm,
            m.Height         AS height_mm,
            m.Price          AS price,
            m.Metadata_URL   AS metadata_url,
            m.Images_URL     AS images_url,
            m.Printsman_Korean AS printsman_korean,
            m.Printsman_Hanja  AS printsman_hanja,
            ph.Name_Korean   AS printshop_db_korean,
            ph.Name_Hanja    AS printshop_db_hanja,
            p.Name_Korean    AS publisher_korean,
            p.Name_Hanja     AS publisher_hanja
        FROM Magazine m
        LEFT JOIN Printing_House ph ON m.Printshop_ID = ph.ID
        LEFT JOIN Publisher p       ON m.Publisher_ID = p.ID
        WHERE m.File_Location IS NOT NULL
        """,
        conn,
    )
    conn.close()
    db["book_dir"] = db["File_Location"].apply(
        lambda p: nfc(p.replace("Database/", "", 1)) if isinstance(p, str) else None
    )
    db = db.dropna(subset=["book_dir"]).drop_duplicates(subset=["book_dir"], keep="first")
    for col in db.select_dtypes(include="object").columns:
        db[col] = db[col].apply(nfc)
    return db


def split_printshop_korean_hanja(combined: str) -> tuple[str | None, str | None]:
    """Split the parquet's combined 'Korean Hanja' printshop name into parts.

    The parquet stores names like '한성도서 주식회사 漢城圖書 株式會社'.
    Hanja and Korean blocks are space-separated. We detect the first hanja
    character and split there.
    """
    if not isinstance(combined, str):
        return None, None
    s = nfc(combined)
    parts = s.split(" ")
    korean_parts, hanja_parts = [], []
    in_hanja = False
    for part in parts:
        if not part:
            continue
        # Heuristic: a token is "hanja" if any of its codepoints is in CJK Unified Ideographs.
        is_hanja = any(0x4E00 <= ord(c) <= 0x9FFF for c in part)
        if is_hanja:
            in_hanja = True
        if in_hanja:
            hanja_parts.append(part)
        else:
            korean_parts.append(part)
    return " ".join(korean_parts) or None, " ".join(hanja_parts) or None


def canonicalize_printshop(raw_korean: str | None) -> str | None:
    if raw_korean is None:
        return None
    return PRINTSHOP_CANON_MAP.get(raw_korean, raw_korean)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path, default=Path("release/dataset_metadata.csv"))
    parser.add_argument("--out-parquet", type=Path, default=Path("release/dataset_metadata.parquet"))
    args = parser.parse_args()

    print(f"Reading parquet: {args.parquet}", file=sys.stderr)
    df = load_parquet(args.parquet)
    print(f"  {len(df)} rows, {df['book_dir'].nunique()} unique books", file=sys.stderr)

    print(f"Reading DB: {args.db}", file=sys.stderr)
    db = load_db(args.db)
    print(f"  {len(db)} unique book records in DB", file=sys.stderr)

    # Split the parquet's combined printshop into raw Korean and Hanja, plus
    # produce the canonical class label that the paper uses.
    pq_split = df["printshop"].apply(split_printshop_korean_hanja)
    df["printshop_raw_korean"] = [a for a, _ in pq_split]
    df["printshop_raw_hanja"] = [b for _, b in pq_split]
    df["printshop_canonical"] = df["printshop_raw_korean"].apply(canonicalize_printshop)

    # Recompute the book_id under the canonical printshop name so that GroupKFold
    # treats both Taedong variants as the same press but still distinguishes
    # books that share a title across presses (e.g. 장한몽 권1).
    df["book_id_canonical"] = df["printshop_canonical"] + "_" + df["book"]

    # Join.
    joined = df.merge(db, on="book_dir", how="left", indicator=True)
    unmatched = (joined["_merge"] == "left_only").sum()
    print(f"Joined: {(joined['_merge'] == 'both').sum()} matched, {unmatched} unmatched", file=sys.stderr)
    if unmatched:
        print("WARNING: some pages did not match a DB record. They will appear with NULL metadata.", file=sys.stderr)
    joined = joined.drop(columns=["_merge", "book_dir", "File_Location"])

    # Final column order. The first columns are the most useful for the paper's
    # GroupKFold workflow; bibliographic columns follow.
    columns = [
        # File identity
        "filename", "image_path", "original_image_path",
        # Class label as used in the paper, plus the raw attribution before merge
        "printshop_canonical", "printshop_raw_korean", "printshop_raw_hanja",
        "printshop_db_korean", "printshop_db_hanja",
        # Book grouping (for GroupKFold)
        "book", "book_id", "book_id_canonical",
        # Bibliographic
        "title_korean", "title_hanja", "title_other",
        "issue_number", "total_issue_number", "frequency", "category",
        "publication_date", "printing_date",
        "page_count_in_volume", "width_mm", "height_mm", "price",
        "publisher_korean", "publisher_hanja",
        "printsman_korean", "printsman_hanja",
        # Source links (Adan Mungo / Hyundam Mun'go)
        "metadata_url", "images_url",
    ]
    out = joined[[c for c in columns if c in joined.columns]].copy()

    # The DB stores empty fields as either empty strings or the literal string
    # 'NaN'. Convert both to actual nulls so the CSV and parquet have consistent
    # missing-value semantics.
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].replace({"": None, "NaN": None, "None": None, "nan": None})

    # Coerce numeric columns to nullable types. The DB stores these as TEXT and
    # often contains empty strings or non-numeric notes alongside actual numbers.
    for col in ("page_count_in_volume",):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in ("width_mm", "height_mm", "price"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Sanity check: per-press counts should match the paper.
    print("\nPer-press page counts (canonical):", file=sys.stderr)
    print(out["printshop_canonical"].value_counts().to_string(), file=sys.stderr)
    print(f"\nUnique book_id_canonical: {out['book_id_canonical'].nunique()}", file=sys.stderr)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    out.to_parquet(args.out_parquet, index=False)
    print(f"\nWrote {args.out_csv} ({args.out_csv.stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    print(f"Wrote {args.out_parquet} ({args.out_parquet.stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
