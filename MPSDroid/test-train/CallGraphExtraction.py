import os
import sys
import time
import zipfile
import argparse
import networkx as nx
from functools import partial
from loguru import logger
from androguard.misc import AnalyzeAPK
from concurrent.futures import ProcessPoolExecutor

logger.remove()
logger.add(sys.stderr, level="WARNING")


def parse_args():
    parser = argparse.ArgumentParser(description='To obtain the call graphs.')
    parser.add_argument(
        '-f', '--file',
        help='输入根目录：递归查找该目录及子目录下的所有 APK 文件进行处理',
        default="/mnt/data5/Obfuscapk/APK/"
    )
    parser.add_argument(
        '-o', '--output',
        help='输出根目录：生成的 .gexf 将按输入目录结构自动分类保存',
        default="/mnt/data5/Temp/"
    )
    parser.add_argument(
        '-j', '--workers',
        type=int,
        default=max(1, min(os.cpu_count() or 1, 80)),
        help='进程数（并发数），默认 CPU 核心数-2'
    )
    parser.add_argument(
        '--delete-on-fail', action='store_true',
        default=True,
        help='遇到解析/生成调用图失败或 APK 非 zip 格式时自动删除该 APK 文件'
    )
    parser.add_argument(
        '--max-process',
        type=int,
        default=5000,
        help='最大“待处理”APK 数量（仅限制需要生成 .gexf 的集合；0 表示不限制）'
    )
    return parser.parse_args()


def get_call_graph(dx):
    CG = nx.DiGraph()
    nodes = dx.find_methods('.*', '.*', '.*', '.*')
    for m in nodes:
        API = m.get_method()
        class_name = API.get_class_name()
        method_name = API.get_name()
        descriptor = API.get_descriptor()
        api_call = class_name + '->' + method_name + descriptor

        if len(m.get_xref_to()) == 0:
            continue
        CG.add_node(api_call)

        for other_class, callee, offset in m.get_xref_to():
            callee_class = callee.get_class_name()
            callee_name = callee.method.get_name()
            callee_desc = callee.method.get_descriptor()
            _callee = callee_class + '->' + callee_name + callee_desc

            CG.add_node(_callee)
            if not CG.has_edge(api_call, _callee):
                CG.add_edge(api_call, _callee)
    return CG


def derive_output_dir(app_path, input_root, out_root):
    rel_dir = os.path.relpath(os.path.dirname(app_path), input_root)
    if rel_dir == '.' or rel_dir == os.curdir:
        return out_root
    return os.path.join(out_root, rel_dir)


def compute_gexf_path(app_path, input_root, out_root, create_parent=False):
    apk_name = os.path.splitext(os.path.basename(app_path))[0]
    target_dir = derive_output_dir(app_path, input_root, out_root)
    if create_parent:
        os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, apk_name + '.gexf')


def apk_to_callgraph(app_path, input_root, out_root, delete_on_fail=False):
    """
    单个 APK 处理：
    - 解析 APK
    - 构建调用图
    - 输出 .gexf

    返回: (消息字符串, 单个 APK 处理耗时秒数)
    """
    apk_name = os.path.splitext(os.path.basename(app_path))[0]

    # 统计单个 APK 耗时
    start_ts = time.time()

    # 非 zip（损坏或伪装）的 APK：根据开关决定是否删除
    if not zipfile.is_zipfile(app_path):
        if delete_on_fail:
            try:
                os.remove(app_path)
                msg = f"NOTZIP_DELETED: {apk_name}"
            except Exception as del_e:
                msg = f"NOTZIP_DELETE_ERROR: {apk_name} | delete_err={del_e}"
        else:
            msg = f"SKIP (not zip): {apk_name}"
        duration = time.time() - start_ts
        return msg, duration

    file_cg = compute_gexf_path(app_path, input_root, out_root, create_parent=True)
    if os.path.exists(file_cg):
        duration = time.time() - start_ts
        return f"EXIST: {apk_name}", duration

    try:
        a, d, dx = AnalyzeAPK(app_path)
        call_graph = get_call_graph(dx=dx)
        nx.write_gexf(call_graph, file_cg)
        msg = f"DONE: {apk_name}"
    except Exception as e:
        if delete_on_fail:
            try:
                os.remove(app_path)
                msg = f"FAIL_DELETED: {apk_name} -> {e}"
            except Exception as del_e:
                msg = f"FAIL_DELETE_ERROR: {apk_name} -> {e} | delete_err={del_e}"
        else:
            msg = f"FAIL: {apk_name} -> {e}"

    duration = time.time() - start_ts
    return msg, duration


def collect_apks(root_dir):
    apks = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.apk'):
                apks.append(os.path.join(dirpath, fn))
    return apks


def main():
    # 在某些环境中使用 spawn 更稳
    try:
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=False)
    except Exception:
        pass

    # -------- 总体开始时间 --------
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))

    args = parse_args()

    out_root = args.output[:-1] if args.output.endswith('/') else args.output
    os.makedirs(out_root, exist_ok=True)

    input_path = args.file
    delete_on_fail = args.delete_on_fail
    max_process = args.max_process

    # 用于统计 APK 平均耗时
    total_apk_time = 0.0   # 所有 APK 的处理总时间（秒）
    processed_count = 0    # 实际发送到 worker 的 APK 数量

    if os.path.isdir(input_path):
        input_root = input_path[:-1] if input_path.endswith('/') else input_path
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

            original_need = len(to_process)
            if max_process > 0 and original_need > max_process:
                to_process = to_process[:max_process]

            print(
                f"Total APKs: {len(apks)} | Already have .gexf: {skipped_existing} | "
                f"Need (before limit): {original_need} | Will process (after limit): {len(to_process)} | "
                f"Limit (--max-process)={max_process} | Delete-on-fail={delete_on_fail}"
            )

            if to_process:
                worker_fn = partial(
                    apk_to_callgraph,
                    input_root=input_root,
                    out_root=out_root,
                    delete_on_fail=delete_on_fail
                )
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    for msg, dur in executor.map(worker_fn, to_process, chunksize=1):
                        if msg:
                            print(msg)
                        total_apk_time += dur
                        processed_count += 1
            else:
                print("Nothing to do. All needed .gexf already exist or limit is 0.")
    else:
        # 单文件模式
        input_root = os.path.dirname(os.path.abspath(input_path)) or '.'
        gexf_path = compute_gexf_path(input_path, input_root, out_root, create_parent=False)
        if os.path.exists(gexf_path):
            print(f"EXIST: {os.path.splitext(os.path.basename(input_path))[0]}")
        else:
            os.makedirs(os.path.dirname(gexf_path), exist_ok=True)
            msg, dur = apk_to_callgraph(
                input_path,
                input_root,
                out_root,
                delete_on_fail=delete_on_fail
            )
            if msg:
                print(msg)
            total_apk_time += dur
            processed_count += 1

    # -------- 结束时间与整体耗时 --------
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts

    print(f"Total elapsed: {elapsed:.2f}s")

    # 计算并打印平均每个 APK 耗时
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

    # 将时间统计写入日志文件（追加）
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
    except Exception as e:
        print(f"WARNING: failed to write time log: {e}")


if __name__ == '__main__':
    main()