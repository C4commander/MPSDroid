import argparse
import os
import re
import sys
from typing import List, Tuple, Dict

import pandas as pd


def slugify_stem(path: str) -> str:
    """从文件路径提取干净的文件名作为后缀，便于区分特征来源。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    # 非字母数字转为下划线，压缩多余下划线
    stem = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_")
    return stem or "src"


def find_column(columns: List[str], target: str) -> str:
    """在列名中查找指定列（不区分大小写），找不到则返回空字符串。"""
    lower_map = {c.lower(): c for c in columns}
    return lower_map.get(target.lower(), "")


def normalize_label(s: pd.Series) -> pd.Series:
    """将标签列转为 0/1 的数值类型；无法解析的设为 NaN。"""
    x = pd.to_numeric(s, errors="coerce")
    # 将非 0/1 的值置为 NaN（例如 2、-1 等）
    x = x.where(x.isin([0, 1]), other=pd.NA)
    return x


def read_prepare_single_csv(
    path: str, sha256_col: str, label_col: str, suffix: str
) -> Tuple[pd.DataFrame, str, str]:
    """
    读取单个 CSV，标准化列名：
    - 只保留 sha256, label 和特征
    - 特征列重命名：原名__{suffix}
    - 同一 CSV 内部若有重复 sha256，只保留第一条
    返回：准备好的 DataFrame 以及最终使用的 sha256/label 列名
    """
    try:
        df = pd.read_csv(path, dtype={sha256_col: str}, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {path} -> {e}")

    # 自动匹配列名（容错大小写）
    sha_col = sha256_col if sha256_col in df.columns else find_column(df.columns.tolist(), sha256_col)
    lab_col = label_col if label_col in df.columns else find_column(df.columns.tolist(), label_col)

    if not sha_col:
        raise ValueError(f"{path} 中未找到 sha256 列（期望: {sha256_col}）")
    if not lab_col:
        raise ValueError(f"{path} 中未找到 label 列（期望: {label_col}）")

    # 规范 sha256 与 label
    df[sha_col] = df[sha_col].astype(str).str.strip()
    df[lab_col] = normalize_label(df[lab_col])

    # 去重：同一 CSV 内部同 sha256 保留第一条
    df = df.drop_duplicates(subset=[sha_col], keep="first")

    # 按列划分
    feature_cols = [c for c in df.columns if c not in (sha_col, lab_col)]

    # 重命名特征列，追加来源后缀
    rename_map = {c: f"{c}__{suffix}" for c in feature_cols}
    df = df.rename(columns=rename_map)

    # 仅保留 sha256, label 和重命名后的特征
    keep_cols = [sha_col, lab_col] + list(rename_map.values())
    df = df[keep_cols]

    # 最终将 sha256/label 标准化命名为统一名称
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
    将多份 CSV 合并：
    - join_policy:
      - inner：以 sha256 的交集合并，要求所有 CSV 都包含该 sha256
      - outer：以 sha256 的并集合并，保留所有出现过的 sha256（可能出现缺失特征）
      - first：保留第一个 CSV 的全部 sha256，其他 CSV 左连接到其上；缺失的特征补 0
    - 每个 CSV 的特征列名追加唯一后缀，避免重名冲突
    - label 列只保留一个，来自多个 CSV 的 label 若冲突会告警；默认优先已有的值，其次取新值
    - 可选：丢弃最终无标签的样本
    - 输出时：将 sha256 放最前、label 放最后
    """
    assert suffix_mode in ("basename", "index")
    assert join_policy in ("inner", "outer", "first")

    prepared_dfs: List[pd.DataFrame] = []
    for idx, path in enumerate(inputs):
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到输入文件: {path}")
        suffix = slugify_stem(path) if suffix_mode == "basename" else f"src{idx+1}"
        cur_df, _, _ = read_prepare_single_csv(path, sha256_col, label_col, suffix)
        prepared_dfs.append(cur_df)

    if not prepared_dfs:
        raise ValueError("没有可用的输入文件。")

    # 计算 sha256 的交集或并集，或保留首文件
    sha_sets = [set(df["sha256"]) for df in prepared_dfs]
    if join_policy == "inner":
        common_sha = set.intersection(*sha_sets) if len(sha_sets) > 1 else sha_sets[0]
        print(f"按交集合并：保留在所有 CSV 中均存在的 sha256，共 {len(common_sha)} 个")
        # 先过滤到交集，避免在合并时产生缺失行
        prepared_dfs = [df[df["sha256"].isin(common_sha)].copy() for df in prepared_dfs]
        how_merge = "inner"
    elif join_policy == "outer":
        print("按并集合并：保留所有出现过的 sha256（可能出现缺失特征）")
        how_merge = "outer"
    else:  # 'first'
        first_set = sha_sets[0]
        print(f"按首文件左连接：保留第一个 CSV 中的所有 sha256，共 {len(first_set)} 个；缺失特征将补 0")
        # 仅过滤后续数据以减少内存（可选，不影响正确性）
        for j in range(1, len(prepared_dfs)):
            prepared_dfs[j] = prepared_dfs[j][prepared_dfs[j]["sha256"].isin(first_set)].copy()
        how_merge = "left"

    # 按序合并
    merged: pd.DataFrame | None = None
    all_conflicts: List[Dict] = []

    for idx, cur_df in enumerate(prepared_dfs):
        if merged is None:
            merged = cur_df
            continue

        # 拆分当前数据的特征与标签，避免合并时出现 label_x/label_y
        cur_features = cur_df.drop(columns=["label"])  # 保留 sha256 + 特征
        cur_labels = cur_df[["sha256", "label"]].rename(columns={"label": "label_new"})

        # 合并特征
        merged = pd.merge(merged, cur_features, on="sha256", how=how_merge)

        # 合并标签并做冲突检测与合并
        merged = pd.merge(merged, cur_labels, on="sha256", how="left")

        # 冲突：两边都有标签且不相等
        mask_both = merged["label"].notna() & merged["label_new"].notna()
        conflict_mask = mask_both & (merged["label"] != merged["label_new"])
        if conflict_mask.any():
            conflicts_df = merged.loc[conflict_mask, ["sha256", "label", "label_new"]].copy()
            conflicts_df["source_file"] = os.path.basename(inputs[idx])
            all_conflicts.append(conflicts_df)

        # 合并标签：已有优先，否则用新值填充
        merged["label"] = merged["label"].combine_first(merged["label_new"])
        merged = merged.drop(columns=["label_new"])

    # 在 first 策略下，将缺失的特征值补 0（仅限特征列，排除 sha256 与 label）
    if join_policy == "first":
        feature_cols = [c for c in merged.columns if c not in ("sha256", "label")]
        if feature_cols:
            merged[feature_cols] = merged[feature_cols].fillna(0)

    # 标签与过滤
    merged["label"] = normalize_label(merged["label"])
    if drop_unlabeled:
        before = len(merged)
        merged = merged[merged["label"].notna()].copy()
        after = len(merged)
        print(f"已丢弃无标签样本: {before - after} 条")

    # 将标签转为可空整数类型（0/1 或 <NA>）
    try:
        merged["label"] = merged["label"].astype("Int64")
    except Exception:
        merged["label"] = pd.to_numeric(merged["label"], errors="coerce")

    # 调整列顺序：sha256 最前，label 最后
    cols_in_df = merged.columns.tolist()
    other_cols = [c for c in cols_in_df if c not in ("sha256", "label")]
    merged = merged[["sha256"] + other_cols + ["label"]]

    # 输出结果
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"合并完成 -> {output}")
    print(f"样本数: {len(merged)}, 特征列数: {len(other_cols)}")

    # 冲突日志
    if all_conflicts:
        conflicts = pd.concat(all_conflicts, ignore_index=True)
        n_conflict_rows = len(conflicts)
        n_conflict_sha = conflicts["sha256"].nunique()
        print(f"警告：发现标签冲突 -> {n_conflict_rows} 条记录，涉及 {n_conflict_sha} 个 sha256")
        if conflict_log:
            conflicts.to_csv(conflict_log, index=False)
            print(f"已输出冲突明细 -> {conflict_log}")
    else:
        print("未发现标签冲突。")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="合并多份含 sha256 与 label 的特征 CSV：确保特征列名唯一；默认按交集（inner）合并。"
    )
    p.add_argument(
        "-i", "--input", nargs="+", help="输入的 CSV 文件列表（两个或以上）",
        default=[
            "./fold_outputs/fold_10/train/api_file_cluster_distribution.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/APIChecker/merge/authority_features.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/APIChecker/merge/harmonic_features.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/APIChecker/merge/pagerank_features.csv",
        ]
    )
    p.add_argument("-o", "--output", default="./fold_outputs/fold_10/train/api_file_cluster_distribution_merge.csv", help="输出合并后的 CSV 路径")
    p.add_argument(
        "--sha256-col", default="sha256", help="各输入 CSV 中表示文件名/ID 的列名（默认: sha256）"
    )
    p.add_argument(
        "--label-col", default="label", help="各输入 CSV 中的标签列名（默认: label，值为 0/1）"
    )
    p.add_argument(
        "--suffix-mode",
        choices=["basename", "index"],
        default="basename",
        help="特征列追加后缀的方式：basename=使用文件名；index=使用序号（默认: basename）"
    )
    p.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="是否丢弃最终无标签的样本（默认: 保留）"
    )
    p.add_argument(
        "--conflict-log",
        default="",
        help="若提供路径，将输出标签冲突明细 CSV（列：sha256,label,label_new,source_file）"
    )
    p.add_argument(
        "--join-policy",
        choices=["inner", "outer", "first"],
        default="first",
        help="合并策略：inner=仅保留所有 CSV 都包含的 sha256；outer=保留并集；first=以第一份为基准左连接（保留首份全部 sha256，缺失特征补 0）"
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    if len(args.input) < 2:
        raise SystemExit("至少需要提供两份输入 CSV。")

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