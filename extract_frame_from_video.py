#!/usr/bin/env python3
"""
并行分段关键帧抽取（避免 pickle/死锁问题，适合长视频）
策略：
- 将视频平均分成 N 段，每段选出 K 帧（贪心选择使帧间差异最大）
- 每个段在独立进程中读取该段帧并返回所选的全局帧索引（仅整数）
- 主进程汇总索引并一次顺序读取/保存选中帧（避免一次性加载所有帧）
优化：
- 在段内计算帧相似度时，用降采样灰度特征向量替代全分辨率，比直接像素差快很多
- 使用 multiprocessing spawn 启动方式，提高 OpenCV 在子进程中稳定性
- 优雅处理 KeyboardInterrupt，尽量关闭子进程
"""
import cv2
import os
import sys
import math
import argparse
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def _frame_feature(gray, downsize=(64, 64)):
    """把灰度帧降采样并展平为 1D 特征向量（float32）——用于快速差异度量"""
    # OpenCV resize expects (width, height)
    small = cv2.resize(gray, downsize, interpolation=cv2.INTER_AREA)
    feat = small.astype(np.float32).ravel() / 255.0
    return feat

def extract_local_indices_from_grays(grays, frames_per_segment, downsize=(64,64)):
    """
    在段内（只接收灰度图列表）用贪心算法选出 frames_per_segment 个局部索引。
    只使用降采样特征计算距离，速度更快。
    返回局部索引列表（相对于段起始为0的索引）。
    """
    n = len(grays)
    if n == 0:
        return []
    if n <= frames_per_segment:
        return list(range(n))

    # 先计算每帧的特征向量（小矩阵）以加速差异计算
    feats = [ _frame_feature(g, downsize=downsize) for g in grays ]

    selected = [0]
    remaining = set(range(1, n))

    # 贪心选择：每次选择与已选帧组中最近距离的最远帧
    while len(selected) < frames_per_segment and remaining:
        max_diff = -1.0
        max_idx = -1
        # 为 speed：将 selected list 的 feat 先取出
        sel_feats = [feats[i] for i in selected]
        for idx in remaining:
            f = feats[idx]
            # compute min distance to selected set (L1 平均)
            # use mean absolute difference for speed/robustness
            diffs = [ np.mean(np.abs(f - s)) for s in sel_feats ]
            mind = min(diffs)
            if mind > max_diff:
                max_diff = mind
                max_idx = idx
        if max_idx == -1:
            break
        selected.append(max_idx)
        remaining.remove(max_idx)

    selected.sort()
    return selected

def process_segment(video_path, start_frame, end_frame, frames_per_segment, downsize=(64,64)):
    """
    在子进程中执行：读取 [start_frame, end_frame) 的帧，计算局部灰度特征并选出局部索引，
    最终返回全局帧索引列表（start_frame + local_index）。
    注意：这是顶层函数（可被 pickle）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"process_segment: cannot open video {video_path}")

    # 尝试设置起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    grays = []
    frames_to_read = end_frame - start_frame
    for _ in range(frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
    cap.release()

    local_sel = extract_local_indices_from_grays(grays, frames_per_segment, downsize=downsize)
    global_sel = [start_frame + idx for idx in local_sel]
    return global_sel

def extract_frames_parallel_safe(video_path, output_dir, num_segments=10, frames_per_segment=10,
                                 downsize=(64,64), max_workers=None):
    """
    主流程入口：
    - 计算分段边界
    - 启动 ProcessPoolExecutor（spawn），并行执行 process_segment
    - 收集所有全局索引、排序去重并写出帧
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取总帧数
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        raise RuntimeError("视频帧数无法读取或为0")

    # 分段
    num_segments = max(1, int(num_segments))
    frames_per_segment = max(1, int(frames_per_segment))
    segment_size = math.floor(total_frames / num_segments)
    segments = []
    for i in range(num_segments):
        s = i * segment_size
        e = (i+1) * segment_size if i < num_segments - 1 else total_frames
        if s >= e:
            continue
        segments.append((s, e))

    # 启动子进程（使用 spawn 更稳健）
    # NOTE: set_start_method 必须在 if __name__ == '__main__' 保护下调用
    if max_workers is None:
        max_workers = min(len(segments), max(1, (os.cpu_count() or 1)))

    print(f"Total frames: {total_frames}, segments: {len(segments)}, segment_size approx: {segment_size}")
    print(f"Workers: {max_workers}, selecting {frames_per_segment} frames per segment.")

    all_indices = []

    # 使用 ProcessPoolExecutor + as_completed，优雅处理 KeyboardInterrupt
    futures = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务（注意：不要传 lambda / 局部函数）
            for (s,e) in segments:
                futures.append(executor.submit(process_segment, video_path, s, e, frames_per_segment, downsize))

            # 收集结果；as_completed 可以在任务完成时逐个获取，避免等待全部完成
            for future in as_completed(futures):
                try:
                    res = future.result()
                    all_indices.extend(res)
                    print(f"Segment done, picked {len(res)} frames, total selected so far {len(all_indices)}")
                except Exception as ex:
                    # 打印但不立刻崩溃：根据场景你可以选择重试或中止
                    print(f"Segment processing error: {ex}", file=sys.stderr)
    except KeyboardInterrupt:
        # 用户中断：尝试取消未完成的 future，然后退出
        print("KeyboardInterrupt received — cancelling pending tasks...", file=sys.stderr)
        for f in futures:
            try:
                f.cancel()
            except Exception:
                pass
        raise

    # 去重并排序
    unique_indices = sorted(set(all_indices))
    print(f"Total unique frames to save: {len(unique_indices)}")

    # 保存选中帧（顺序读取视频，只写需要的帧）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot reopen video for saving frames")

    saved = 0
    cur = 0
    indices_set = set(unique_indices)
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices_set:
            out_path = os.path.join(output_dir, f"frame_{saved:05d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1
            # optional: remove from set to speed membership if many frames are saved
            # indices_set.remove(idx)
        if (idx+1) % 500 == 0:
            print(f"Scanned {idx+1}/{total_frames} frames, saved {saved}")
    cap.release()
    print(f"Done. Saved {saved} frames into {output_dir}")

def parse_args():
    p = argparse.ArgumentParser(description="Parallel segment keyframe extractor (robust, no-deadlock)")
    p.add_argument("video", help="Path to input video")
    p.add_argument("--output", "-o", default="frames", help="Output directory")
    p.add_argument("--segments", type=int, default=10, help="Number of segments")
    p.add_argument("--per_segment", type=int, default=10, help="Frames per segment")
    p.add_argument("--downsize", type=int, default=64, help="Downsize (square) for feature extraction. Larger = more accurate but slower")
    p.add_argument("--workers", type=int, default=None, help="Number of worker processes (default = min(segments, CPU))")
    return p.parse_args()

if __name__ == "__main__":
    # 使用 spawn 启动子进程，避免 fork 导致的 OpenCV/Python 交互问题
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # 如果已经设置过了就忽略
        pass

    args = parse_args()
    extract_frames_parallel_safe(
        video_path=args.video,
        output_dir=args.output,
        num_segments=args.segments,
        frames_per_segment=args.per_segment,
        downsize=(args.downsize, args.downsize),
        max_workers=args.workers
    )
