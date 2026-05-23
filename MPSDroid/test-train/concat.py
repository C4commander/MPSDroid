import argparse
import os
import re
import sys
from typing import List, Tuple, Dict

import pandas as pd


def slugify_stem(path: str) -> str:
    """Extract a clean file name from the path as a suffix, convenient for feature source identification."""
    stem = os.path.splitext(os.path.basename(path))[0]
    # Convert non-alphanumeric to underscore, compress multiple underscores
    stem = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_")
    return stem or "src"


def find_column(columns: List[str], target: str) -> str:
    """Find the specified column in a list of columns (case-insensitive), return empty string if not found."""
    lower_map = {c.lower(): c for c in columns}
    return lower_map.get(target.lower(), "")


def normalize_label(s: pd.Series) -> pd.Series:
    """Convert label column to numerical 0/1; unparseable values set as NaN."""
    x = pd.to_numeric(s, errors="coerce")
    # Set anything not 0/1 as NaN (e.g. 2, -1, etc.)
    x = x.where(x.isin([0, 1]), other=pd.NA)
    return x


def read_prepare_single_csv(
    path: str, sha256_col: str, label_col: str, suffix: str
) -> Tuple[pd.DataFrame, str, str]:
    """
    Read a single CSV and standardize column names:
    - Keep only sha256, label, and features
    - Rename feature columns: original_name__{suffix}
    - If duplicate sha256 in one CSV, keep the first
    Returns: prepared DataFrame and the final sha256/label column names
    """
    try:
        df = pd.read_csv(path, dtype={sha256_col: str}, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {path} -> {e}")

    # Automatically match column names (case-insensitive)
    sha_col = sha256_col if sha256_col in df.columns else find_column(df.columns.tolist(), sha256_col)
    lab_col = label_col if label_col in df.columns else find_column(df.columns.tolist(), label_col)

    if not sha_col:
        raise ValueError(f"sha256 column not found in {path} (expected: {sha256_col})")
    if not lab_col:
        raise ValueError(f"label column not found in {path} (expected: {label_col})")

    # Standardize sha256 and label
    df[sha_col] = df[sha_col].astype(str).str.strip()
    df[lab_col] = normalize_label(df[lab_col])

    # Remove duplicates: keep first for duplicate sha256 in the same CSV
    df = df.drop_duplicates(subset=[sha_col], keep="first")

    # Split by column
    feature_cols = [c for c in df.columns if c not in (sha_col, lab_col)]

    # Rename feature columns with a unique suffix for the source
    rename_map = {c: f"{c}__{suffix}" for c in feature_cols}
    df = df.rename(columns=rename_map)

    # Only keep sha256, label, and renamed features
    keep_cols = [sha_col, lab_col] + list(rename_map.values())
    df = df[keep_cols]

    # Finally, standardize sha256/label column names
    df = df.rename(columns={sha_col: "sha256", lab_col: "label"})

    return df, "sha256", "label"


def merge_many_csvs(
    inputs: List[str],
    output: str,
    sha256_col: str = "sha256",
    label_col: str = "label",
    suffix_mode: str = "basename",
    drop_unlabeled: bool = False,
    conflict_log: str = "",
    join_policy: str = "inner",  # inner/outer/first
) -> None:
    """
    Merge multiple CSVs:
    - join_policy:
      - inner: merge on the intersection of sha256, requiring sha256 in all CSVs
      - outer: merge on the union of sha256, keeps all sha256 ever present (may have missing features)
      - first: keep all sha256 from the first CSV, left-join the others; missing features fill with 0
    - For each CSV, feature columns have unique suffixes to avoid naming conflicts
    - Only one label column is kept; if conflicting labels, print warnings; existing value takes precedence, otherwise use the new one
    - Optionally: drop samples without any label in the result
    - Output: sha256 is first column, label is last
    """
    assert suffix_mode in ("basename", "index")
    assert join_policy in ("inner", "outer", "first")

    prepared_dfs: List[pd.DataFrame] = []
    for idx, path in enumerate(inputs):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        suffix = slugify_stem(path) if suffix_mode == "basename" else f"src{idx+1}"
        cur_df, _, _ = read_prepare_single_csv(path, sha256_col, label_col, suffix)
        prepared_dfs.append(cur_df)

    if not prepared_dfs:
        raise ValueError("No valid input files.")

    # Calculate sha256 intersection/union, or keep first file
    sha_sets = [set(df["sha256"]) for df in prepared_dfs]
    if join_policy == "inner":
        common_sha = set.intersection(*sha_sets) if len(sha_sets) > 1 else sha_sets[0]
        print(f"Merging by intersection: keep sha256 that are in all CSV, total {len(common_sha)}")
        # Filter to intersection first to avoid missing rows during merge
        prepared_dfs = [df[df["sha256"].isin(common_sha)].copy() for df in prepared_dfs]
        how_merge = "inner"
    elif join_policy == "outer":
        print("Merging by union: keep all sha256 ever present (may result in missing features).")
        how_merge = "outer"
    else:  # 'first'
        first_set = sha_sets[0]
        print(f"Merging left on first file: keep all sha256 in the first CSV, total {len(first_set)}; missing features will be 0")
        # Optionally filter later files for less memory usage (optional, does not impact correctness)
        for j in range(1, len(prepared_dfs)):
            prepared_dfs[j] = prepared_dfs[j][prepared_dfs[j]["sha256"].isin(first_set)].copy()
        how_merge = "left"

    # Merge in sequence
    merged: pd.DataFrame | None = None
    all_conflicts: List[Dict] = []

    for idx, cur_df in enumerate(prepared_dfs):
        if merged is None:
            merged = cur_df
            continue

        # Split current data's features and label to avoid label_x/label_y in merge
        cur_features = cur_df.drop(columns=["label"])  # Keep sha256 + features only
        cur_labels = cur_df[["sha256", "label"]].rename(columns={"label": "label_new"})

        # Merge features
        merged = pd.merge(merged, cur_features, on="sha256", how=how_merge)

        # Merge labels and detect label conflicts
        merged = pd.merge(merged, cur_labels, on="sha256", how="left")

        # Conflict: both sides have label, but not identical
        mask_both = merged["label"].notna() & merged["label_new"].notna()
        conflict_mask = mask_both & (merged["label"] != merged["label_new"])
        if conflict_mask.any():
            conflicts_df = merged.loc[conflict_mask, ["sha256", "label", "label_new"]].copy()
            conflicts_df["source_file"] = os.path.basename(inputs[idx])
            all_conflicts.append(conflicts_df)

        # Merge labels: existing has priority, else use new value
        merged["label"] = merged["label"].combine_first(merged["label_new"])
        merged = merged.drop(columns=["label_new"])

    # In 'first' strategy, fill missing features with 0 (features only, not sha256/label)
    if join_policy == "first":
        feature_cols = [c for c in merged.columns if c not in ("sha256", "label")]
        if feature_cols:
            merged[feature_cols] = merged[feature_cols].fillna(0)

    # Label and filter
    merged["label"] = normalize_label(merged["label"])
    if drop_unlabeled:
        before = len(merged)
        merged = merged[merged["label"].notna()].copy()
        after = len(merged)
        print(f"Dropped unlabeled samples: {before - after}")

    # Cast label as nullable integer (0/1 or <NA>)
    try:
        merged["label"] = merged["label"].astype("Int64")
    except Exception:
        merged["label"] = pd.to_numeric(merged["label"], errors="coerce")

    # Reorder columns: sha256 first, label last
    cols_in_df = merged.columns.tolist()
    other_cols = [c for c in cols_in_df if c not in ("sha256", "label")]
    merged = merged[["sha256"] + other_cols + ["label"]]

    # Output results
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Merging done -> {output}")
    print(f"Sample count: {len(merged)}, feature columns: {len(other_cols)}")

    # Conflict log
    if all_conflicts:
        conflicts = pd.concat(all_conflicts, ignore_index=True)
        n_conflict_rows = len(conflicts)
        n_conflict_sha = conflicts["sha256"].nunique()
        print(f"Warning: found label conflicts -> {n_conflict_rows} rows, involving {n_conflict_sha} sha256")
        if conflict_log:
            conflicts.to_csv(conflict_log, index=False)
            print(f"Conflict details written to -> {conflict_log}")
    else:
        print("No label conflicts found.")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge multiple CSVs with sha256 and label: ensure feature column names are unique. Merge by intersection (inner) by default."
    )
    p.add_argument(
        "-i", "--input", nargs="+", help="List of input CSV files (at least two required)",
        default=[
            "./statistic/train/file_cluster_distribution.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/test-train/APIChecker-ob/result/train/authority_features.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/test-train/APIChecker-ob/result/train/harmonic_features.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/test-train/APIChecker-ob/result/train/pagerank_features.csv",
        ]
    )
    p.add_argument("-o", "--output", default="/mnt/data2/wb2024/Methodology/MPSDroid/test-train/statistic/train/file_cluster_distribution_merge.csv", help="Output merged CSV file path")
    p.add_argument(
        "--sha256-col", default="sha256", help="Column in input CSV representing file name/ID (default: sha256)"
    )
    p.add_argument(
        "--label-col", default="label", help="Label column in input CSV (default: label, values: 0/1)"
    )
    p.add_argument(
        "--suffix-mode",
        choices=["basename", "index"],
        default="basename",
        help="How to suffix feature column names: basename=use file name; index=use index (default: basename)"
    )
    p.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Whether to drop samples without final label (default: keep them)"
    )
    p.add_argument(
        "--conflict-log",
        default="",
        help="If provided, output label conflicts to CSV (columns: sha256,label,label_new,source_file)"
    )
    p.add_argument(
        "--join-policy",
        choices=["inner", "outer", "first"],
        default="first",
        help="Merge strategy: inner=keep sha256 in all CSVs only; outer=keep union; first=left join using first file (keep all sha256 in primary, fill missing features with 0)"
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    if len(args.input) < 2:
        raise SystemExit("At least two input CSVs are required.")

    merge_many_csvs(
        inputs=args.input,
        output=args.output,
        sha256_col=args.sha256_col,
        label_col=args.label_col,
        suffix_mode=args.suffix_mode,
        drop_unlabeled=args.drop_unlabeled,
        conflict_log=args.conflict_log,
        join_policy=args.join_policy,
    )


if __name__ == "__main__":
    main(sys.argv[1:])