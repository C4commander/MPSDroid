import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier  # 新增：KNN

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
            colsample_bytree=0.8,  # 0.8
            reg_lambda=1,  # 0.8
            n_jobs=-1,
            n_estimators=300,  # 300
            learning_rate=0.08,  # 0.08
            max_depth=6,  # 6
            subsample=0.8,  # 0.8
            eval_metric='logloss'
        )
    elif model_type == 'knn':
        # 新增：KNN 模型，这里给一个比较常见的默认配置
        # 如果你有特定需求，可以修改 n_neighbors、weights 等参数
        return KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',   # 或 'uniform'
            metric='minkowski',   # p=2 即欧氏距离
            p=2
        )
    else:
        raise ValueError("model_type 必须为 'rf'、'et'、'gbdt'、'xgb' 或 'knn'")


def nanmean_fmt(arr):
    arr = np.array(arr, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")


def determine_feature_columns(df_tr: pd.DataFrame, df_te: pd.DataFrame, quiet: bool = False):
    """
    新的特征列逻辑：
    - 假定第一列是 sha256（不作为特征）
    - 假定最后一列是 label（不作为特征）
    - 其它全部列作为特征
    兼容健壮性：
    - 若 label 不在最后一列，依旧排除列名为 label
    - 若第一列不是 sha256 也依旧按“位置第一列”排除；并给出提示
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

    # 也确保 sha256 不被误包含（如果不是第一列但出现于中间）
    feat_cols = [c for c in feat_cols if c.lower() != "sha256"]

    if not feat_cols:
        raise ValueError("未能得到任何特征列（除去第一与最后之后为空）。")

    if not quiet:
        print(f"[INFO] 使用特征列数量: {len(feat_cols)}，示例前5个: {feat_cols[:5]}")
    return feat_cols


def eval_one_fold(train_csv: str, test_csv: str, seed: int, model_type: str, quiet: bool = False):
    df_tr = pd.read_csv(train_csv, encoding="utf-8-sig")
    df_te = pd.read_csv(test_csv, encoding="utf-8-sig")

    if "label" not in df_tr.columns or "label" not in df_te.columns:
        raise KeyError(f"缺少 label 列: {train_csv} 或 {test_csv}")

    # 仅保留标签 0/1
    df_tr = df_tr[df_tr["label"].isin([0, 1])].copy()
    df_te = df_te[df_te["label"].isin([0, 1])].copy()

    # 新的特征列逻辑
    feat_cols = determine_feature_columns(df_tr, df_te, quiet=quiet)

    # 测试集与训练特征对齐：缺的补0，多的丢弃
    for c in feat_cols:
        if c not in df_te.columns:
            df_te[c] = 0
    X_tr = df_tr[feat_cols].values
    y_tr = df_tr["label"].values.astype(int)
    # 只保留训练确定的列顺序
    X_te = df_te[feat_cols].values
    y_te = df_te["label"].values.astype(int)

    # 若训练集单一类别跳过
    uniq_tr = np.unique(y_tr)
    if uniq_tr.size < 2:
        if not quiet:
            print(f"[WARN] 训练集仅包含单一类别 {uniq_tr.tolist()}，跳过该折: {train_csv}")
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


def external_folds_evaluation(folds_root: str, seed: int, model_type: str, quiet: bool = False):
    folds = sorted([p for p in glob.glob(os.path.join(folds_root, "fold_*")) if os.path.isdir(p)])
    if not folds:
        raise FileNotFoundError(f"未找到折目录: {folds_root}/fold_*")

    if not quiet:
        print(f"发现 {len(folds)} 折，开始外部折评估...")

    per_fold = []
    for fdir in folds:
        # train_csv = os.path.join(fdir, "train", "file_cluster_distribution_merge.csv")
        # test_csv = os.path.join(fdir, "test", "file_cluster_distribution_merge.csv")
        train_csv = os.path.join(fdir, "train", "file_cluster_distribution.csv")
        test_csv = os.path.join(fdir, "test", "file_cluster_distribution.csv")
        if not os.path.isfile(train_csv) or not os.path.isfile(test_csv):
            if not quiet:
                print(f"[WARN] 缺少 train/test 分布CSV，跳过该折: {fdir}")
            continue

        metrics = eval_one_fold(train_csv, test_csv, seed, model_type, quiet=quiet)
        if metrics is None:
            per_fold.append({
                "fold": os.path.basename(fdir),
                "status": "skipped_single_class_train"
            })
            continue

        row = {"fold": os.path.basename(fdir), "status": "ok"}
        row.update(metrics)
        per_fold.append(row)

        if not quiet:
            print(
                f"{row['fold']}: "
                f"ACC={row['ACC']:.4f}, "
                f"TPR={row['TPR']:.4f}, TNR={row['TNR']:.4f}, FPR={row['FPR']:.4f}, FNR={row['FNR']:.4f}, "
                f"Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1']:.4f}"
            )

    ok_rows = [r for r in per_fold if r.get("status") == "ok"]
    avg = {}
    if ok_rows:
        for k in ["ACC", "TPR", "TNR", "FPR", "FNR", "Precision", "Recall", "F1"]:
            avg[k] = nanmean_fmt([r[k] for r in ok_rows])

    return per_fold, avg


def main():
    parser = argparse.ArgumentParser(description="每折输出进行外部折测试与平均汇总（特征=除第一与最后列外全部列）")
    parser.add_argument("--folds-root", default="/mnt/data2/wb2024/Methodology/MyWay2.0/fold_outputs-mc", help="包含 fold_01, fold_02 等目录的根路径")
    parser.add_argument("--model-type", choices=["rf", "et", "gbdt", "xgb", "knn"], default="xgb", help="分类模型类型") 
    parser.add_argument("--random-state", type=int, default=42, help="随机种子（对树模型有效，knn 无随机性）")
    parser.add_argument("--report-json", type=str, default=None, help="评估结果JSON路径（默认 folds-root/model_eval_summary_merge.json）")
    parser.add_argument("--report-csv", type=str, default=None, help="逐折评估结果CSV路径（默认 folds-root/model_eval_per_fold_merge.csv）")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    args = parser.parse_args()

    per_fold, avg = external_folds_evaluation(
        folds_root=args.folds_root,
        seed=args.random_state,
        model_type=args.model_type,
        quiet=args.quiet
    )

    if avg:
        print("\n平均指标:")
        print(
            f"F1={avg.get('F1', float('nan')):.4f}, "
            f"Precision={avg.get('Precision', float('nan')):.4f}, "
            f"Recall={avg.get('Recall', float('nan')):.4f}, "
            f"ACC={avg.get('ACC', float('nan')):.4f}, "
            f"TPR={avg.get('TPR', float('nan')):.4f}, "
            f"TNR={avg.get('TNR', float('nan')):.4f}, "
            f"FPR={avg.get('FPR', float('nan')):.4f}, "
            f"FNR={avg.get('FNR', float('nan')):.4f}"
        )
    else:
        print("\n无可用折用于计算平均（可能全部被跳过）。")

    folds_root = args.folds_root
    if args.report_json is None:
        args.report_json = os.path.join(folds_root, "model_eval_summary_merge.json")
    if args.report_csv is None:
        args.report_csv = os.path.join(folds_root, "model_eval_per_fold_merge.csv")

    try:
        summary = {
            "folds_root": folds_root,
            "model_type": args.model_type,
            "random_state": args.random_state,
            "per_fold": per_fold,
            "average": avg
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"评估摘要JSON写入: {args.report_json}")
    except Exception as e:
        print(f"[WARN] 写入JSON失败: {e}")

    try:
        ok_rows = [r for r in per_fold if r.get("status") == "ok"]
        if ok_rows:
            pd.DataFrame(ok_rows).to_csv(args.report_csv, index=False, encoding="utf-8-sig")
            print(f"逐折评估结果CSV写入: {args.report_csv}")
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