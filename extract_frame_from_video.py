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

    # 先计算每帧的特征向量矩阵 (n, d)
    feats = np.stack([_frame_feature(g, downsize=downsize) for g in grays], axis=0)

    # 初始化
    selected = [0]
    remaining = np.ones(n, dtype=bool)
    remaining[0] = False

    # 初始 min_dists
    min_dists = np.full(n, np.inf, dtype=np.float32)
    last_feat = feats[0]

    while len(selected) < frames_per_segment and remaining.any():
        # 计算当前已选帧到所有点的距离，并更新最小距离
        dists = np.mean(np.abs(feats - last_feat), axis=1)
        min_dists = np.minimum(min_dists, dists)

        # 在未选中的帧中找到最远点
        min_dists_masked = np.where(remaining, min_dists, -1.0)
        next_idx = int(np.argmax(min_dists_masked))
        if min_dists_masked[next_idx] < 0:
            break

        selected.append(next_idx)
        remaining[next_idx] = False
        last_feat = feats[next_idx]

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

def extract_frames_parallel_safe(video_path, output_dir, num_segments=10, frames_per_segment=10, max_edge=512,
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
            h, w = frame.shape[:2]
            long_side = max(w, h)
            scale = max_edge / long_side

            new_w = int(w * scale)
            new_h = int(h * scale)

            # 缩放
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 保存
            out_path = os.path.join(output_dir, f"frame_{saved:05d}.png")
            cv2.imwrite(out_path, resized)
            saved += 1
            # optional: remove from set to speed membership if many frames are saved
            # indices_set.remove(idx)
    cap.release()
    print(f"Done. Saved {saved} frames into {output_dir}")

def parse_args():
    p = argparse.ArgumentParser(description="Parallel segment keyframe extractor (robust, no-deadlock)")
    p.add_argument("video", help="Path to input video")
    p.add_argument("--output", "-o", default="frames", help="Output directory")
    p.add_argument("--segments", type=int, default=30, help="Number of segments")
    p.add_argument("--per_segment", type=int, default=1, help="Frames per segment")
    p.add_argument("--max_edge", type=int, default=518, help="Max edge size for saved frames")
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
    if args.max_edge != 518:
        print(f"Warning: max_edge is not 518, not consistent with default VGGT settings.")
    extract_frames_parallel_safe(
        video_path=args.video,
        output_dir=args.output,
        num_segments=args.segments,
        frames_per_segment=args.per_segment,
        max_edge=args.max_edge,
        downsize=(args.downsize, args.downsize),
        max_workers=args.workers
    )
