import cv2
import os
import numpy as np
import argparse


def extract_frames(video_path, output_dir, use_motion_check=False, motion_threshold=30):
    """
    从视频中抽帧
    :param video_path: 输入视频路径
    :param output_dir: 输出帧保存目录
    :param use_motion_check: 是否使用运动检测去重帧
    :param motion_threshold: 运动检测阈值
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"Video info: FPS={fps}, Total frames={frame_count}, Duration={duration:.2f}s")

    prev_gray = None
    saved_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        save_frame = True
        if use_motion_check:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
                if diff < motion_threshold:
                    save_frame = False
            prev_gray = gray

        if save_frame:
            frame_file = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(frame_file, frame)
            saved_frames += 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames, saved {saved_frames} frames")

    cap.release()
    print(f"Done! Total frames saved: {saved_frames}")


if __name__ == "__main__":
    """
    ✅ 使用方法
    
    抽取所有帧：
    python extract_frames.py input_video.mp4 --output frames
    
    
    使用运动检测去重（可减少重复帧）：
    python extract_frames.py input_video.mp4 --output frames --motion --threshold 25
    
    --threshold 越小 → 越严格，帧差小就会被丢掉
    默认保存 PNG，避免运动伪影和压缩模糊
    """
    parser = argparse.ArgumentParser(description="Video Frame Extractor")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="frames", help="Directory to save frames")
    parser.add_argument("--motion", action="store_true", help="Enable motion detection to skip similar frames")
    parser.add_argument("--threshold", type=float, default=30.0, help="Motion threshold for frame difference")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.motion, args.threshold)
