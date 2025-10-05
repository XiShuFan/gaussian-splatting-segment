import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import argparse

def extract_segment_frames(segment_frames, segment_grays, frames_per_segment):
    """
    对一段帧进行贪心抽帧
    :param segment_frames: 彩色帧列表
    :param segment_grays: 灰度帧列表
    :param frames_per_segment: 每段抽取帧数
    :return: 选中的帧索引（相对于段内）
    """
    if len(segment_frames) <= frames_per_segment:
        return list(range(len(segment_frames)))

    selected = [0]  # 初始选择第一帧
    remaining = set(range(1, len(segment_frames)))

    while len(selected) < frames_per_segment and remaining:
        max_diff = -1
        max_idx = -1
        for idx in remaining:
            diff = min(np.mean(np.abs(segment_grays[idx].astype(float) - segment_grays[i].astype(float)))
                       for i in selected)
            if diff > max_diff:
                max_diff = diff
                max_idx = idx
        selected.append(max_idx)
        remaining.remove(max_idx)
    return selected

def process_segment(video_path, start_frame, end_frame, frames_per_segment):
    """
    处理视频段
    :return: 相对于整段视频的帧索引
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    segment_frames = []
    segment_grays = []

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        segment_frames.append(frame)
        segment_grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    selected_local = extract_segment_frames(segment_frames, segment_grays, frames_per_segment)
    # 转换为全局索引
    selected_global = [start_frame + idx for idx in selected_local]
    return selected_global

def extract_frames_parallel(video_path, output_dir, num_segments=10, frames_per_segment=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    segment_size = total_frames // num_segments
    segments = [(i * segment_size, (i + 1) * segment_size if i < num_segments - 1 else total_frames)
                for i in range(num_segments)]

    # 并行处理
    all_indices = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(lambda seg: process_segment(video_path, seg[0], seg[1], frames_per_segment), segments)
        for segment_indices in results:
            all_indices.extend(segment_indices)

    # 按顺序保存帧
    cap = cv2.VideoCapture(video_path)
    saved = 0
    current_idx = 0
    all_indices_set = set(all_indices)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in all_indices_set:
            cv2.imwrite(os.path.join(output_dir, f"frame_{saved:05d}.png"), frame)
            saved += 1
        current_idx += 1
    cap.release()
    print(f"Done! Saved {saved} frames in {output_dir}")

if __name__ == "__main__":
    """
    1.不一次性加载全部帧：视频按段读取，避免内存占用过大。
    2.并行处理每段：利用 concurrent.futures.ProcessPoolExecutor。
    3.每段内部用帧差异贪心选择：保证运动变化大的帧被保留。
    4.可控总帧数：总帧数 = num_segments * frames_per_segment。
    """
    parser = argparse.ArgumentParser(description="Video Frame Extractor (Segment Parallel)")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="frames", help="Directory to save frames")
    parser.add_argument("--segments", type=int, default=10, help="Number of segments")
    parser.add_argument("--per_segment", type=int, default=10, help="Frames per segment")
    args = parser.parse_args()

    extract_frames_parallel(args.video, args.output, args.segments, args.per_segment)
