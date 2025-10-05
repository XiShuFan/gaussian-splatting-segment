import cv2
import os
import numpy as np
import argparse

def extract_frames_max_diff(video_path, output_dir, target_frame_count=100):
    """
    从视频中抽取指定数量的帧，保证帧差最大
    :param video_path: 视频路径
    :param output_dir: 输出帧目录
    :param target_frame_count: 希望抽取的帧数
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    # 读取所有帧并转为灰度
    frames = []
    grays = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        grays.append(gray)
        frame_idx += 1

    cap.release()
    total_frames = len(frames)
    print(f"Total frames: {total_frames}")

    # 初始化：先保留第一帧
    selected_indices = [0]
    remaining = set(range(1, total_frames))

    # 使用贪心方法选择帧差最大的帧
    while len(selected_indices) < target_frame_count and remaining:
        max_diff = -1
        max_idx = -1
        for idx in remaining:
            # 与已选帧中最相似的帧比较差异
            diff = min(np.mean(np.abs(grays[idx].astype(float) - grays[i].astype(float))) for i in selected_indices)
            if diff > max_diff:
                max_diff = diff
                max_idx = idx
        selected_indices.append(max_idx)
        remaining.remove(max_idx)
        if len(selected_indices) % 10 == 0:
            print(f"Selected {len(selected_indices)}/{target_frame_count} frames...")

    # 保存选中的帧
    for i, idx in enumerate(selected_indices):
        frame_file = os.path.join(output_dir, f"frame_{i:05d}.png")
        cv2.imwrite(frame_file, frames[idx])

    print(f"Done! Selected {len(selected_indices)} frames saved in {output_dir}")

if __name__ == "__main__":
    """
    基于“帧差优先”的抽帧策略
    1.先将视频全部帧读取到内存并转换为灰度
    2.用贪心算法，每次选择与已选帧差异最大的帧
    3.保证最终抽取的 target_frame_count 帧之间尽可能不同
    4.输出 PNG 文件，避免压缩导致的运动伪影
    """
    parser = argparse.ArgumentParser(description="Video Frame Extractor with Max Difference")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="frames", help="Directory to save frames")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to extract")
    args = parser.parse_args()

    extract_frames_max_diff(args.video, args.output, args.num_frames)
