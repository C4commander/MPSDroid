import argparse
import os
import time
import traceback
import zipfile
from multiprocessing import Pool as ThreadPool
from pathlib import Path

import networkx as nx
from androguard.misc import AnalyzeAPK

from loguru import logger
import sys
try:
    import psutil
except Exception:
    psutil = None

try:
    import resource
except Exception:
    resource = None

logger.remove()
logger.add(sys.stderr, level="WARNING")

_MB = 1024.0 * 1024.0


def get_rss_mb():
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / _MB
    except Exception:
        return None


def get_peak_rss_mb():
    if resource is None:
        return None
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return peak / _MB
        return peak / 1024.0
    except Exception:
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='To obtain the call graphs.')
    parser.add_argument('-f', '--file', default="/mnt/data2/wb2024/Data/data-md", help='The path of an APK file or a dir contains some APK files', type=str)
    parser.add_argument('-o', '--output', default="/mnt/data2/wb2024/Methodology/MyWay/data/Graph-md", help='The path of output.', type=str)
    parser.add_argument('-w', '--workers', help='Number of APKs processed in parallel when input is a directory', type=int, default=120)
    args = parser.parse_args()
    return args

def iter_method_analyses(dx):
    """Iterate over MethodAnalysis objects across older/newer androguard releases."""
    if hasattr(dx, "get_methods"):
        return dx.get_methods()
    return dx.find_methods('.*', '.*', '.*', '.*')

def unwrap_method(method_like):
    """Return the underlying method object for MethodAnalysis / ExternalMethod / EncodedMethod."""
    if hasattr(method_like, "get_method"):
        return method_like.get_method()
    return method_like

def build_method_signature(method_like):
    method_obj = unwrap_method(method_like)
    return (
        method_obj.get_class_name()
        + '->'
        + method_obj.get_name()
        + method_obj.get_descriptor()
    )

def get_call_graph(dx):
    CG = nx.DiGraph()
    for m in iter_method_analyses(dx):
        api_call = build_method_signature(m)

        xrefs = list(m.get_xref_to())
        if len(xrefs) == 0:
            continue
        CG.add_node(api_call)

        for _, callee, _ in xrefs:
            _callee = build_method_signature(callee)
            CG.add_node(_callee)
            if not CG.has_edge(api_call, _callee):
                CG.add_edge(api_call, _callee)

    return CG

def resolve_output_gexf(apk_path: Path, output_root: Path, input_root: Path | None):
    if input_root is not None:
        relative_apk = apk_path.relative_to(input_root)
        return output_root / relative_apk.with_suffix('.gexf')
    return output_root / f"{apk_path.stem}.gexf"

def apk_to_callgraph(app_path, output_root, input_root=None):
    started_at = time.time()
    apk_path = Path(app_path)
    apk_name = apk_path.stem
    output_root = Path(output_root)
    input_root_path = Path(input_root) if input_root is not None else None
    file_cg = resolve_output_gexf(apk_path, output_root, input_root_path)

    if file_cg.exists():
        return {"apk": str(apk_path), "status": "skipped_existing", "elapsed": 0.0, "output": str(file_cg)}
    if not zipfile.is_zipfile(app_path):
        return {"apk": str(apk_path), "status": "skipped_invalid_zip", "elapsed": 0.0}

    try:
        # AnalyzeAPK remains the supported high-level entrypoint in current androguard
        # releases, while dx exposes modern Analysis / MethodAnalysis accessors.
        _, _, dx = AnalyzeAPK(str(apk_path))
        rss_start_mb = get_rss_mb()
        graph_started_at = time.time()
        cg = get_call_graph(dx=dx)
        graph_elapsed = time.time() - graph_started_at
        rss_end_mb = get_rss_mb()

        file_cg.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(cg, file_cg)

        elapsed = graph_elapsed
        return {
            "apk": str(apk_path),
            "status": "ok",
            "elapsed": elapsed,
            "nodes": cg.number_of_nodes(),
            "edges": cg.number_of_edges(),
            "output": str(file_cg),
            "rss_start_mb": rss_start_mb,
            "rss_end_mb": rss_end_mb,
            "rss_delta_mb": None if rss_start_mb is None or rss_end_mb is None else rss_end_mb - rss_start_mb,
            "peak_rss_mb": get_peak_rss_mb(),
            "total_elapsed": time.time() - started_at,
        }
    except Exception as exc:
        elapsed = time.time() - started_at
        rss_start_mb = get_rss_mb()
        rss_end_mb = get_rss_mb()
        return {
            "apk": str(apk_path),
            "status": "error",
            "elapsed": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "rss_start_mb": rss_start_mb,
            "rss_end_mb": rss_end_mb,
            "rss_delta_mb": None if rss_start_mb is None or rss_end_mb is None else rss_end_mb - rss_start_mb,
            "peak_rss_mb": get_peak_rss_mb(),
            "total_elapsed": elapsed,
        }

def _apk_worker(args):
    return apk_to_callgraph(*args)

def log_result(result, processed_count, processed_elapsed, total_count):
    apk_name = result["apk"]
    status = result["status"]
    elapsed = result["elapsed"]
    progress = f"[{processed_count}/{total_count}]"

    if status == "ok":
        avg = processed_elapsed / processed_count if processed_count else 0.0
        print(
            f"{progress} [OK] {apk_name} | elapsed={elapsed:.4f}s | avg={avg:.4f}s/apk | "
            f"nodes={result.get('nodes', 0)} | edges={result.get('edges', 0)}"
        )
    elif status == "error":
        avg = processed_elapsed / processed_count if processed_count else 0.0
        print(f"{progress} [ERROR] {apk_name} | elapsed={elapsed:.4f}s | avg={avg:.4f}s/apk | {result['error']}")
        print(result["traceback"])
    else:
        print(f"{progress} [SKIP] {apk_name} | reason={status}")

def main():
    tic = time.time()
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    out_path = Path(args.output)

    if os.path.isdir(args.file):
        input_root = Path(args.file).resolve()
        apks = sorted(path for path in input_root.rglob('*') if path.is_file() and path.suffix.lower() == '.apk')
        total_count = len(apks)
        pool = ThreadPool(args.workers)
        processed_count = 0
        processed_elapsed = 0.0
        total_memory_mb = 0.0
        memory_count = 0
        ok_count = 0
        error_count = 0
        skip_count = 0
        finished_count = 0

        for result in pool.imap_unordered(
            _apk_worker,
            [(str(apk), str(out_path), str(input_root)) for apk in apks],
        ):
            finished_count += 1
            if result["status"] == "ok":
                processed_count += 1
                processed_elapsed += result["elapsed"]
                ok_count += 1
            elif result["status"] == "error":
                processed_count += 1
                processed_elapsed += result["elapsed"]
                error_count += 1
            else:
                skip_count += 1
            if result.get("rss_end_mb") is not None:
                total_memory_mb += result["rss_end_mb"]
                memory_count += 1
            log_result(result, finished_count, processed_elapsed, total_count)

        pool.close()
        pool.join()
    else:
        result = apk_to_callgraph(args.file, out_path)
        total_count = 1
        processed_count = 1 if result["status"] in {"ok", "error"} else 0
        processed_elapsed = result["elapsed"] if processed_count else 0.0
        total_memory_mb = result["rss_end_mb"] if result.get("rss_end_mb") is not None else 0.0
        memory_count = 1 if result.get("rss_end_mb") is not None else 0
        log_result(result, 1, processed_elapsed, total_count)
        ok_count = 1 if result["status"] == "ok" else 0
        error_count = 1 if result["status"] == "error" else 0
        skip_count = 1 if result["status"].startswith("skipped_") else 0

    total_elapsed = time.time() - tic
    avg_elapsed = processed_elapsed / processed_count if processed_count else 0.0
    avg_memory_mb = total_memory_mb / memory_count if memory_count else 0.0
    print(
        f"[SUMMARY] total_elapsed={total_elapsed:.4f}s | processed={processed_count} | "
        f"ok={ok_count} | error={error_count} | skipped={skip_count} | "
        f"avg_call_graph={avg_elapsed:.4f}s/apk | avg_memory={avg_memory_mb:.4f}MB"
    )

if __name__ == '__main__':
    main()

