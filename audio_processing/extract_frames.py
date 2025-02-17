# pigs_health_recognition/src/video_processing/extract_frames.py

import cv2
import os


def extract_frames(video_path, output_dir, fps=1):
    """
    从视频中提取帧并保存为图像文件。

    :param video_path: 视频文件路径
    :param output_dir: 输出帧的保存目录
    :param fps: 提取帧的频率（每秒提取多少帧）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"保存帧: {frame_filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"总共保存了 {saved_count} 帧。")


if __name__ == "__main__":
    video_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/video/202309151632.mp4'  # 更新为您的视频路径
    output_dir = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/frames'
    extract_frames(video_path, output_dir, fps=1)  # 每秒提取1帧
