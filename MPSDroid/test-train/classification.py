import os
import json
import argparse
import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier  # KNN

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def select_model(model_type: str, seed: int):
    if model_type == 'rf':
        return RandomForestClassifier(
            random_state=seed,
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
    elif model_type == 'et':
        return ExtraTreesClassifier(
            random_state=seed,
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
    elif model_type == 'gbdt':
        return GradientBoostingClassifier(
            random_state=seed,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=1.0
        )
    elif model_type == 'xgb':
        if not HAS_XGB:
            raise ImportError("xgboost 未安装，请选择其他模型或安装 xgboost")
        return xgb.XGBClassifier(
            random_state=seed,
            colsample_bytree=0.8,
            reg_lambda=1,
            n_jobs=-1,
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            eval_metric='logloss'
        )
    elif model_type == 'knn':
        return KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            metric='minkowski',
            p=2
        )
    else:
        raise ValueError("model_type 必须为 'rf'、'et'、'gbdt'、'xgb' 或 'knn'")


def determine_feature_columns(df_tr: pd.DataFrame, df_te: pd.DataFrame, quiet: bool = False):
    """
    特征列逻辑：
    - 假定第一列是 sha256（不作为特征）
    - 假定最后一列是 label（不作为特征）
    - 其它全部列作为特征
    兼容健壮性：
    - 若 label 不在最后一列，依旧排除列名为 label
    - 若第一列不是 sha256 也依旧按“位置第一列”排除，并提示
    """
    cols_tr = list(df_tr.columns)

    if "label" not in cols_tr:
        raise KeyError("训练集中缺少 label 列")
    if "label" not in df_te.columns:
        raise KeyError("测试集中缺少 label 列")

    if len(cols_tr) < 3:
        raise ValueError("列数不足：至少需要 sha256、一个特征列、label")

    first_col = cols_tr[0]
    last_col = cols_tr[-1]

    if not quiet:
        if first_col.lower() != "sha256":
            print(f"[WARN] 第一列名称不是 sha256，而是 '{first_col}'，仍按位置排除第一列")
        if last_col != "label":
            print(f"[WARN] 最后一列名称不是 label，而是 '{last_col}'，将另外排除真正的 label 列")

    # 基于位置排除第一与最后一列
    feat_cols = cols_tr[1:-1]

    # 若 label 不在最后一列，额外排除
    if last_col != "label" and "label" in feat_cols:
        feat_cols = [c for c in feat_cols if c != "label"]

    # 也确保 sha256 不被误包含
    feat_cols = [c for c in feat_cols if c.lower() != "sha256"]

    if not feat_cols:
        raise ValueError("未能得到任何特征列（除去第一与最后之后为空）。")

    if not quiet:
        print(f"[INFO] 使用特征列数量: {len(feat_cols)}，示例前5个: {feat_cols[:5]}")
    return feat_cols


def eval_one_split(train_csv: str, test_csv: str, seed: int, model_type: str, quiet: bool = False):
    df_tr = pd.read_csv(train_csv, encoding="utf-8-sig")
    df_te = pd.read_csv(test_csv, encoding="utf-8-sig")

    if "label" not in df_tr.columns or "label" not in df_te.columns:
        raise KeyError(f"缺少 label 列: {train_csv} 或 {test_csv}")

    # 仅保留标签 0/1
    df_tr = df_tr[df_tr["label"].isin([0, 1])].copy()
    df_te = df_te[df_te["label"].isin([0, 1])].copy()

    # 获取特征列
    feat_cols = determine_feature_columns(df_tr, df_te, quiet=quiet)

    # 对齐特征列（缺的补0，多的丢弃）
    for c in feat_cols:
        if c not in df_te.columns:
            df_te[c] = 0
    X_tr = df_tr[feat_cols].values
    y_tr = df_tr["label"].values.astype(int)
    X_te = df_te[feat_cols].values
    y_te = df_te["label"].values.astype(int)

    uniq_tr = np.unique(y_tr)
    if uniq_tr.size < 2:
        if not quiet:
            print(f"[WARN] 训练集仅包含单一类别 {uniq_tr.tolist()}，无法训练有效分类器")
        return None

    clf = select_model(model_type, seed)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        TN = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        FP = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        FN = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        TP = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    else:
        TN, FP, FN, TP = cm.ravel()

    tpr = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    tnr = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    fpr = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    fnr = FN / (FN + TP) if (FN + TP) > 0 else np.nan
    acc = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, zero_division=0)
    recall = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)

    return {
        "ACC": float(acc), "TPR": float(tpr), "TNR": float(tnr), "FPR": float(fpr), "FNR": float(fnr),
        "Precision": float(precision), "Recall": float(recall), "F1": float(f1)
    }


def main():
    parser = argparse.ArgumentParser(
        description="在指定目录下的 train/test 分布文件上评估分类器（不再使用交叉验证）。"
    )
    parser.add_argument(
        "--root-dir",
        default="./statistic",
        help="包含 train/ 与 test/ 子目录的根目录；其下有 "
             "train/file_cluster_distribution.csv 和 test/file_cluster_distribution.csv"
    )
    parser.add_argument(
        "--model-type",
        choices=["rf", "et", "gbdt", "xgb", "knn"],
        default="xgb",
        help="分类模型类型"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子（对树模型有效，knn 无随机性）"
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="评估结果JSON路径（默认 root-dir/model_eval_summary.json）"
    )
    parser.add_argument(
        "--report-csv",
        type=str,
        default=None,
        help="评估结果CSV路径（默认 root-dir/model_eval_single.csv）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出"
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    train_csv = os.path.join(root_dir, "train", "file_cluster_distribution_merge.csv")
    test_csv = os.path.join(root_dir, "test", "file_cluster_distribution_merge.csv")

    if not os.path.isfile(train_csv) or not os.path.isfile(test_csv):
        raise FileNotFoundError(
            f"未找到 train/test 分布CSV：\n  train: {train_csv}\n  test : {test_csv}"
        )

    if not args.quiet:
        print(f"[INFO] 使用根目录: {root_dir}")
        print(f"[INFO] 训练文件: {train_csv}")
        print(f"[INFO] 测试文件: {test_csv}")
        print(f"[INFO] 模型类型: {args.model_type}, random_state={args.random_state}")

    metrics = eval_one_split(
        train_csv=train_csv,
        test_csv=test_csv,
        seed=args.random_state,
        model_type=args.model_type,
        quiet=args.quiet
    )

    per_split = []
    if metrics is None:
        status = "skipped_single_class_train"
        if not args.quiet:
            print("[WARN] 训练集仅包含单一类别，本次评估被跳过。")
        per_split.append({
            "root": os.path.basename(os.path.normpath(root_dir)),
            "status": status
        })
        avg = {}
    else:
        status = "ok"
        row = {"root": os.path.basename(os.path.normpath(root_dir)), "status": status}
        row.update(metrics)
        per_split.append(row)
        avg = metrics

        if not args.quiet:
            print("\n评估结果:")
            print(
                f"ACC={row['ACC']:.4f}, "
                f"TPR={row['TPR']:.4f}, TNR={row['TNR']:.4f}, "
                f"FPR={row['FPR']:.4f}, FNR={row['FNR']:.4f}, "
                f"Precision={row['Precision']:.4f}, "
                f"Recall={row['Recall']:.4f}, "
                f"F1={row['F1']:.4f}"
            )

    if args.report_json is None:
        args.report_json = os.path.join(root_dir, "model_eval_summary.json")
    if args.report_csv is None:
        args.report_csv = os.path.join(root_dir, "model_eval_single.csv")

    # 写 JSON
    try:
        summary = {
            "root_dir": root_dir,
            "model_type": args.model_type,
            "random_state": args.random_state,
            "result": per_split[0] if per_split else {},
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"评估摘要JSON写入: {args.report_json}")
    except Exception as e:
        print(f"[WARN] 写入JSON失败: {e}")

    # 写 CSV（单行）
    try:
        ok_rows = [r for r in per_split if r.get("status") == "ok"]
        if ok_rows:
            pd.DataFrame(ok_rows).to_csv(args.report_csv, index=False, encoding="utf-8-sig")
            if not args.quiet:
                print(f"评估结果CSV写入: {args.report_csv}")
    except Exception as e:
        print(f"[WARN] 写入CSV失败: {e}")


if __name__ == "__main__":
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    main()
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")