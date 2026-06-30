import argparse
import gc
import sys
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from xml.sax.saxutils import escape
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract APK function call graphs as GEXF files.")
    parser.add_argument(
        "-f", "--file",
        help="Input root directory: recursively process all APK files under this directory.",
        default="/mnt/data2/wb2024/Data/data-md",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output root directory: generated .gexf files preserve the input directory structure.",
        default="/mnt/data2/wb2024/Methodology/MyWay/data/Graph-md",
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=max(1, min(os.cpu_count() or 1, 120)),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--delete-on-fail",
        action="store_true",
        default=False,
        help="Delete APK files that cannot be parsed or are not valid zip files.",
    )
    parser.add_argument(
        "--max-process",
        type=int,
        default=5000,
        help="Maximum number of APKs to process; 0 means no limit.",
    )
    parser.add_argument(
        "--huge-apk-mb",
        type=float,
        default=12.0,
        help="APKs at or above this size are processed with reduced concurrency to lower peak RAM.",
    )
    parser.add_argument(
        "--large-apk-mb",
        type=float,
        default=6.0,
        help="APKs at or above this size are processed with moderately reduced concurrency.",
    )
    parser.add_argument(
        "--huge-apk-workers",
        type=int,
        default=4,
        help="Worker cap for huge APKs.",
    )
    parser.add_argument(
        "--large-apk-workers",
        type=int,
        default=12,
        help="Worker cap for large APKs.",
    )
    return parser.parse_args()


def method_signature(method):
    return f"{method.get_class_name()}->{method.get_name()}{method.get_descriptor()}"


def collect_call_graph_data(dx):
    node_ids = {}
    node_names = []
    edges = set()

    def intern_node(name):
        idx = node_ids.get(name)
        if idx is None:
            idx = len(node_names)
            node_ids[name] = idx
            node_names.append(name)
        return idx

    for method_analysis in dx.find_methods(".*", ".*", ".*", ".*"):
        xrefs = method_analysis.get_xref_to()
        if not xrefs:
            continue

        caller = method_signature(method_analysis.get_method())
        caller_idx = intern_node(caller)

        for _other_class, callee, _offset in xrefs:
            callee_sig = method_signature(callee.method)
            callee_idx = intern_node(callee_sig)
            edges.add((caller_idx, callee_idx))
    return node_names, edges


def write_gexf_fast(node_names, edges, output_path):
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        handle.write('<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n')
        handle.write('  <graph mode="static" defaultedgetype="directed">\n')
        handle.write('    <nodes>\n')
        for node_name in node_names:
            escaped = escape(node_name, {'"': "&quot;"})
            handle.write(f'      <node id="{escaped}" label="{escaped}" />\n')
        handle.write('    </nodes>\n')
        handle.write('    <edges>\n')
        for edge_id, (src_idx, dst_idx) in enumerate(edges):
            src = escape(node_names[src_idx], {'"': "&quot;"})
            dst = escape(node_names[dst_idx], {'"': "&quot;"})
            handle.write(f'      <edge id="{edge_id}" source="{src}" target="{dst}" />\n')
        handle.write('    </edges>\n')
        handle.write('  </graph>\n')
        handle.write('</gexf>\n')


def normalize_root(path):
    return path[:-1] if path.endswith(("/", "\\")) else path


def derive_output_dir(app_path, input_root, out_root):
    rel_dir = os.path.relpath(os.path.dirname(app_path), input_root)
    if rel_dir == "." or rel_dir == os.curdir:
        return out_root
    return os.path.join(out_root, rel_dir)


def compute_gexf_path(app_path, input_root, out_root, create_parent=False):
    apk_name = os.path.splitext(os.path.basename(app_path))[0]
    target_dir = derive_output_dir(app_path, input_root, out_root)
    if create_parent:
        os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, apk_name + ".gexf")


def collect_apks(root_dir):
    apks = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".apk"):
                apks.append(os.path.join(dirpath, fn))
    return sorted(apks)


def sort_apks_for_balanced_scheduling(apk_paths):
    def sort_key(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return -1

    return sorted(apk_paths, key=sort_key, reverse=True)


def split_tasks_by_size(tasks, large_apk_mb, huge_apk_mb):
    huge = []
    large = []
    normal = []
    for apk_path in tasks:
        try:
            size_mb = os.path.getsize(apk_path) / (1024.0 * 1024.0)
        except OSError:
            size_mb = 0.0
        if size_mb >= huge_apk_mb:
            huge.append(apk_path)
        elif size_mb >= large_apk_mb:
            large.append(apk_path)
        else:
            normal.append(apk_path)
    return huge, large, normal


def maybe_delete(path, enabled, ok_msg, fail_msg):
    if not enabled:
        return ok_msg
    try:
        os.remove(path)
        return ok_msg.replace("SKIP", "NOTZIP_DELETED").replace("FAIL", "FAIL_DELETED")
    except Exception as exc:
        return f"{fail_msg} | delete_err={exc}"


def apk_to_callgraph(app_path, input_root, out_root, delete_on_fail=False):
    apk_name = os.path.splitext(os.path.basename(app_path))[0]
    start_ts = time.time()
    analysis = None
    node_names = None
    edges = None

    if not zipfile.is_zipfile(app_path):
        msg = maybe_delete(
            app_path,
            delete_on_fail,
            f"SKIP (not zip): {apk_name}",
            f"NOTZIP_DELETE_ERROR: {apk_name}",
        )
        return msg, time.time() - start_ts

    file_cg = compute_gexf_path(app_path, input_root, out_root, create_parent=True)
    if os.path.exists(file_cg):
        return f"EXIST: {apk_name}", time.time() - start_ts

    try:
        from androguard.misc import AnalyzeAPK

        _apk, _dex, analysis = AnalyzeAPK(app_path)
        node_names, edges = collect_call_graph_data(analysis)
        write_gexf_fast(node_names, edges, file_cg)
        msg = f"DONE: {apk_name}"
    except Exception as exc:
        msg = maybe_delete(
            app_path,
            delete_on_fail,
            f"FAIL: {apk_name} -> {exc}",
            f"FAIL_DELETE_ERROR: {apk_name} -> {exc}",
        )
    finally:
        del analysis, node_names, edges
        gc.collect()

    return msg, time.time() - start_ts


def process_batches(tasks, input_root, out_root, args):
    total_apk_time = 0.0
    processed_count = 0
    worker_fn = partial(
        apk_to_callgraph,
        input_root=input_root,
        out_root=out_root,
        delete_on_fail=args.delete_on_fail,
    )

    huge_tasks, large_tasks, normal_tasks = split_tasks_by_size(
        tasks,
        large_apk_mb=args.large_apk_mb,
        huge_apk_mb=args.huge_apk_mb,
    )
    batches = [
        ("huge", huge_tasks, max(1, min(args.workers, args.huge_apk_workers))),
        ("large", large_tasks, max(1, min(args.workers, args.large_apk_workers))),
        ("normal", normal_tasks, max(1, args.workers)),
    ]

    for batch_name, batch_tasks, batch_workers in batches:
        if not batch_tasks:
            continue
        print(
            f"Starting {batch_name} APK batch: count={len(batch_tasks)} | workers={batch_workers}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=batch_workers) as executor:
            for msg, dur in executor.map(worker_fn, batch_tasks, chunksize=1):
                if msg:
                    print(msg)
                total_apk_time += dur
                processed_count += 1
    return total_apk_time, processed_count


def main():
    try:
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=False)
    except Exception:
        pass

    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))

    args = parse_args()
    out_root = normalize_root(args.output)
    os.makedirs(out_root, exist_ok=True)

    input_path = args.file
    delete_on_fail = args.delete_on_fail
    max_process = args.max_process
    total_apk_time = 0.0
    processed_count = 0

    if os.path.isdir(input_path):
        input_root = normalize_root(input_path)
        apks = collect_apks(input_root)

        if not apks:
            print(f"No APK files found under: {input_root}")
        else:
            skipped_existing = 0
            to_process = []
            for apk in apks:
                gexf_path = compute_gexf_path(apk, input_root, out_root, create_parent=False)
                if os.path.exists(gexf_path):
                    skipped_existing += 1
                else:
                    to_process.append(apk)

            to_process = sort_apks_for_balanced_scheduling(to_process)
            original_need = len(to_process)
            if max_process > 0 and original_need > max_process:
                to_process = to_process[:max_process]

            print(
                f"Total APKs: {len(apks)} | Already have .gexf: {skipped_existing} | "
                f"Need (before limit): {original_need} | Will process (after limit): {len(to_process)} | "
                f"Limit (--max-process)={max_process} | Workers={args.workers} | Delete-on-fail={delete_on_fail}"
            )

            if to_process:
                total_apk_time, processed_count = process_batches(to_process, input_root, out_root, args)
            else:
                print("Nothing to do. All needed .gexf already exist or limit is 0.")
    else:
        input_root = os.path.dirname(os.path.abspath(input_path)) or "."
        gexf_path = compute_gexf_path(input_path, input_root, out_root, create_parent=False)
        if os.path.exists(gexf_path):
            print(f"EXIST: {os.path.splitext(os.path.basename(input_path))[0]}")
        else:
            os.makedirs(os.path.dirname(gexf_path), exist_ok=True)
            msg, dur = apk_to_callgraph(
                input_path,
                input_root,
                out_root,
                delete_on_fail=delete_on_fail,
            )
            if msg:
                print(msg)
            total_apk_time += dur
            processed_count += 1

    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts

    print(f"Total elapsed: {elapsed:.2f}s")

    if processed_count > 0:
        avg_per_apk = total_apk_time / processed_count
        print(
            f"Processed APKs: {processed_count} | "
            f"Total APK compute time: {total_apk_time:.2f}s | "
            f"Average per APK: {avg_per_apk:.2f}s"
        )
    else:
        avg_per_apk = 0.0
        print("Processed APKs: 0")

    log_path = os.path.join(out_root, "run_time.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Start: {start_str} | End: {end_str} | "
                f"TotalElapsed: {elapsed:.2f}s | "
                f"ProcessedAPKs: {processed_count} | "
                f"TotalAPKTime: {total_apk_time:.2f}s | "
                f"AvgPerAPK: {avg_per_apk:.2f}s | "
                f"Input: {input_path}\n"
            )
    except Exception as exc:
        print(f"WARNING: failed to write time log: {exc}")


if __name__ == "__main__":
    main()
