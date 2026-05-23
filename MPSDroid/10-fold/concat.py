import argparse
import os
import re
import sys
from typing import List, Tuple, Dict

import pandas as pd


def slugify_stem(path: str) -> str:
    """Extract a clean file stem from the file path for use as a suffix, to distinguish feature sources."""
    stem = os.path.splitext(os.path.basename(path))[0]
    # Replace non-alphanumeric characters with underscores, compress repeated underscores
    stem = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_")
    return stem or "src"


def find_column(columns: List[str], target: str) -> str:
    """Find the given column (case-insensitive) in the column names, return empty string if not found."""
    lower_map = {c.lower(): c for c in columns}
    return lower_map.get(target.lower(), "")


def normalize_label(s: pd.Series) -> pd.Series:
    """Convert label column to numeric 0/1 type; anything unparsable is set to NaN."""
    x = pd.to_numeric(s, errors="coerce")
    # Set non-0/1 values as NaN (such as 2, -1, etc.)
    x = x.where(x.isin([0, 1]), other=pd.NA)
    return x


def read_prepare_single_csv(
    path: str, sha256_col: str, label_col: str, suffix: str
) -> Tuple[pd.DataFrame, str, str]:
    """
    Read a single CSV and standardize column names:
    - Only keep sha256, label, and features
    - Feature columns are renamed: original__{suffix}
    - For duplicate sha256 rows only the first row is kept
    Return: prepared DataFrame and resulting sha256/label column names
    """
    try:
        df = pd.read_csv(path, dtype={sha256_col: str}, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {path} -> {e}")

    # Auto match column names (tolerant, case-insensitive)
    sha_col = sha256_col if sha256_col in df.columns else find_column(df.columns.tolist(), sha256_col)
    lab_col = label_col if label_col in df.columns else find_column(df.columns.tolist(), label_col)

    if not sha_col:
        raise ValueError(f"sha256 column not found in {path} (expected: {sha256_col})")
    if not lab_col:
        raise ValueError(f"label column not found in {path} (expected: {label_col})")

    # Standardize sha256 and label
    df[sha_col] = df[sha_col].astype(str).str.strip()
    df[lab_col] = normalize_label(df[lab_col])

    # Drop duplicates: keep first for each sha256 in the same CSV
    df = df.drop_duplicates(subset=[sha_col], keep="first")

    # Separate feature columns
    feature_cols = [c for c in df.columns if c not in (sha_col, lab_col)]

    # Rename feature columns, append source suffix
    rename_map = {c: f"{c}__{suffix}" for c in feature_cols}
    df = df.rename(columns=rename_map)

    # Only keep sha256, label, and renamed features
    keep_cols = [sha_col, lab_col] + list(rename_map.values())
    df = df[keep_cols]

    # Standardize sha256/label to a uniform name
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
    Merge multiple CSV files:
    - join_policy:
      - inner: intersect sha256, only keep rows present in all CSVs
      - outer: union of sha256, keep all sha256 ever present (may have missing features)
      - first: keep all sha256 from the first CSV, left-join the others to it; missing features are filled as 0
    - Each CSV's feature column names get a unique suffix to avoid name collision
    - Only one label column is kept; if labels differ across inputs, a warning is produced; existing values are prioritized, otherwise use new
    - Option: drop samples with no final label
    - Output: sha256 column first, label column last
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
        raise ValueError("No valid input files found.")

    # Compute intersection/union of sha256 or keep the first file
    sha_sets = [set(df["sha256"]) for df in prepared_dfs]
    if join_policy == "inner":
        common_sha = set.intersection(*sha_sets) if len(sha_sets) > 1 else sha_sets[0]
        print(f"Merging by intersection: keeping sha256s present in all inputs, count: {len(common_sha)}")
        # Filter to intersection before merging to avoid missing rows
        prepared_dfs = [df[df["sha256"].isin(common_sha)].copy() for df in prepared_dfs]
        how_merge = "inner"
    elif join_policy == "outer":
        print("Merging by union: keep all sha256 found in any input (may have missing features)")
        how_merge = "outer"
    else:  # 'first'
        first_set = sha_sets[0]
        print(f"Merging left on first input: keep all sha256 from first CSV, count: {len(first_set)}; missing features will be 0")
        # Optionally filter later data for memory efficiency (doesn't affect correctness)
        for j in range(1, len(prepared_dfs)):
            prepared_dfs[j] = prepared_dfs[j][prepared_dfs[j]["sha256"].isin(first_set)].copy()
        how_merge = "left"

    # Sequential merging
    merged: pd.DataFrame | None = None
    all_conflicts: List[Dict] = []

    for idx, cur_df in enumerate(prepared_dfs):
        if merged is None:
            merged = cur_df
            continue

        # Split current data's features and label (avoid label_x/label_y columns)
        cur_features = cur_df.drop(columns=["label"])  # keep sha256 + features only
        cur_labels = cur_df[["sha256", "label"]].rename(columns={"label": "label_new"})

        # Merge features
        merged = pd.merge(merged, cur_features, on="sha256", how=how_merge)

        # Merge labels and detect label conflicts
        merged = pd.merge(merged, cur_labels, on="sha256", how="left")

        # Conflict: both sides have non-na labels, but not equal
        mask_both = merged["label"].notna() & merged["label_new"].notna()
        conflict_mask = mask_both & (merged["label"] != merged["label_new"])
        if conflict_mask.any():
            conflicts_df = merged.loc[conflict_mask, ["sha256", "label", "label_new"]].copy()
            conflicts_df["source_file"] = os.path.basename(inputs[idx])
            all_conflicts.append(conflicts_df)

        # Merge label: prefer existing, else fill with new
        merged["label"] = merged["label"].combine_first(merged["label_new"])
        merged = merged.drop(columns=["label_new"])

    # For 'first' strategy, fill missing feature values with 0 (features only, not sha256/label)
    if join_policy == "first":
        feature_cols = [c for c in merged.columns if c not in ("sha256", "label")]
        if feature_cols:
            merged[feature_cols] = merged[feature_cols].fillna(0)

    # Label filtering
    merged["label"] = normalize_label(merged["label"])
    if drop_unlabeled:
        before = len(merged)
        merged = merged[merged["label"].notna()].copy()
        after = len(merged)
        print(f"Dropped samples with no label: {before - after}")

    # Cast label to nullable integer type (0/1 or <NA>)
    try:
        merged["label"] = merged["label"].astype("Int64")
    except Exception:
        merged["label"] = pd.to_numeric(merged["label"], errors="coerce")

    # Move columns: sha256 first, label last
    cols_in_df = merged.columns.tolist()
    other_cols = [c for c in cols_in_df if c not in ("sha256", "label")]
    merged = merged[["sha256"] + other_cols + ["label"]]

    # Output result
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Merged file -> {output}")
    print(f"Sample count: {len(merged)}, number of feature columns: {len(other_cols)}")

    # Conflict log
    if all_conflicts:
        conflicts = pd.concat(all_conflicts, ignore_index=True)
        n_conflict_rows = len(conflicts)
        n_conflict_sha = conflicts["sha256"].nunique()
        print(f"Warning: label conflicts detected -> {n_conflict_rows} records in {n_conflict_sha} unique sha256")
        if conflict_log:
            conflicts.to_csv(conflict_log, index=False)
            print(f"Conflict details written to -> {conflict_log}")
    else:
        print("No label conflicts found.")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge multiple feature CSVs with sha256 and label: guarantee unique feature names; default merge is intersection (inner)."
    )
    p.add_argument(
        "-i", "--input", nargs="+", help="List of input CSV files (at least two required)",
        default=[
            "./fold_outputs/fold_10/train/api_file_cluster_distribution.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/APIChecker/merge/authority_features.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/APIChecker/merge/harmonic_features.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/APIChecker/merge/pagerank_features.csv",
        ]
    )
    p.add_argument("-o", "--output", default="./fold_outputs/fold_10/train/api_file_cluster_distribution_merge.csv", help="Path for the merged output CSV")
    p.add_argument(
        "--sha256-col", default="sha256", help="Column name representing filename/ID in input CSVs (default: sha256)"
    )
    p.add_argument(
        "--label-col", default="label", help="Label column in input CSV (default: label, value should be 0/1)"
    )
    p.add_argument(
        "--suffix-mode",
        choices=["basename", "index"],
        default="basename",
        help="How to suffix feature columns: basename=use filename; index=use source index (default: basename)"
    )
    p.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Whether to drop samples with no final label (default: keep them)"
    )
    p.add_argument(
        "--conflict-log",
        default="",
        help="If set, output label conflict details CSV (columns: sha256,label,label_new,source_file)"
    )
    p.add_argument(
        "--join-policy",
        choices=["inner", "outer", "first"],
        default="first",
        help="Merge strategy: inner=only keep sha256 present in all inputs; outer=union; first=left join on first (keep all sha256 present in first input, fill missing features with 0)"
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