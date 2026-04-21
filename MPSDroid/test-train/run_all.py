#!/usr/bin/env python3
import os
import sys
import subprocess
import concurrent.futures as cf
import time
TASKS = [

    ["python", "Laplace_smoothing.py", "--test-dir", "./Sequences/ResStringEncryption"],
    ["python", "sequences_merge.py", "--test-data-root", "./Sequences/ResStringEncryption"],
    ["python", "clustering.py"],
    ["python", "concat.py", "--input", "./statistic/train/file_cluster_distribution.csv",
        "/mnt/data2/wb2024/Methodology/MalScan/test-train/APIChecker-ob/result/train/authority_features.csv",
        "/mnt/data2/wb2024/Methodology/MalScan/test-train/APIChecker-ob/result/train/harmonic_features.csv",
        "/mnt/data2/wb2024/Methodology/MalScan/test-train/APIChecker-ob/result/train/pagerank_features.csv",
        "--output", 
        "./statistic/train/file_cluster_distribution_merge.csv"],
    ["python", "concat.py", "--input", 
            "./statistic/test/file_cluster_distribution.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/test-train/APIChecker-ob/result/ResStringEncryption/authority_features.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/test-train/APIChecker-ob/result/ResStringEncryption/harmonic_features.csv",
            "/mnt/data2/wb2024/Methodology/MalScan/test-train/APIChecker-ob/result/ResStringEncryption/pagerank_features.csv",
            "--output", 
            "./statistic/test/file_cluster_distribution_merge.csv"],
    ["python", "classification.py"],

    # ["python", "CV2.py", "-d", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/merge", "-o", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/result", "-t", "degree"],
    # ["python", "CV2.py", "-d", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/merge", "-o", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/result", "-t", "katz"],
    # ["python", "CV2.py", "-d", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/merge", "-o", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/result", "-t", "closeness"],
    # ["python", "CV2.py", "-d", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/merge", "-o", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/result", "-t", "harmonic"],
    # ["python", "CV2.py", "-d", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/merge", "-o", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/result", "-t", "pagerank"],
    # ["python", "CV2.py", "-d", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/merge", "-o", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/result", "-t", "eigenvector"],
    # ["python", "CV2.py", "-d", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/merge", "-o", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/result", "-t", "authority"],
    # ["python", "CV2.py", "-d", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/merge", "-o", "/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/result"]
]

MAX_WORKERS = 1
LOG_DIR = "logs"


def run_task(idx: int, args: list[str]) -> tuple[int, int, float]:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"job_{idx}.log")

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"[JOB {idx}] Command: {' '.join(args)}\n")
        fh.write(f"[JOB {idx}] Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}\n")
        fh.write("-" * 60 + "\n")
        fh.flush()

        proc = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT)
        rc = proc.wait()

    end = time.time()
    duration = end - start

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\n" + "-" * 60 + "\n")
        fh.write(f"[JOB {idx}] End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}\n")
        fh.write(f"[JOB {idx}] Duration: {duration:.2f} seconds\n")

    return idx, rc, duration


def main() -> int:
    failures = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_task, i, t): i for i, t in enumerate(TASKS)}
        for fut in cf.as_completed(futs):
            i, rc, duration = fut.result()
            print(f"[{i}] Exit code: {rc}, Duration: {duration:.2f} seconds")
            if rc != 0:
                failures.append(i)
    if failures:
        print(f"Failed jobs: {failures}. Check {LOG_DIR}/job_<idx>.log")
        return 1
    print("All jobs done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
