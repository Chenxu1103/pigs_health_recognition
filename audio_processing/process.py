# /Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/audio_processing/process.py
import cv2
from PIL import Image
import librosa
import soundfile as sf
import os
import ffmpeg  # 使用 ffmpeg 进行音频提取

def extract_audio_from_video(video_path, output_audio_path):
    """
    从视频文件中提取音频并保存为 WAV 文件
    :param video_path: 视频文件路径
    :param output_audio_path: 输出音频文件路径
    """
    try:
        # 使用 ffmpeg 提取音频
        os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {output_audio_path}")
        print(f"Audio extracted and saved to {output_audio_path}")
    except Exception as e:
        print(f"Error extracting audio from video: {e}")


# 音频文件加载
def load_audio_files(audio_path):
    """
    加载音频文件
    :param audio_path: 音频文件路径
    :return: 音频数据和采样率
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, None


# 音频特征提取：提取 MFCC 特征
def extract_features(y, sr):
    """
    提取音频特征，使用 MFCC
    :param y: 音频数据
    :param sr: 采样率
    :return: MFCC 特征
    """
    if y is None or sr is None:
        return None

    # 计算 MFCC 特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc


# 视频处理：使用 OpenCV 读取视频帧
def process_video(video_path, output_folder):
    """
    处理视频，将视频帧保存到指定文件夹
    :param video_path: 视频文件路径
    :param output_folder: 输出文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:  # 每10帧保存一次
            # 转换为 PIL 图像并保存
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
            img.save(frame_filename)
            print(f"Saved {frame_filename}")

    cap.release()
    print("Video processing complete.")


# 音频处理：使用 librosa 提取音频特征并保存音频文件
def process_audio(audio_path, output_path):
    """
    处理音频文件，提取特征并保存处理后的音频
    :param audio_path: 音频文件路径
    :param output_path: 处理后的音频文件保存路径
    """
    # 读取音频文件
    y, sr = load_audio_files(audio_path)

    if y is None or sr is None:
        print("Audio loading failed, skipping audio processing.")
        return

    # 提取音频特征（例如 MFCC）
    mfcc = extract_features(y, sr)
    if mfcc is not None:
        print(f"MFCC shape: {mfcc.shape}")

    # 将音频保存为新的 WAV 文件
    sf.write(output_path, y, sr)
    print(f"Audio saved to {output_path}")


# 主函数
def main():
    # 路径配置
    video_path = '/pigs_health_recognition/data/video/202309151308（疑似）.mp4'  # 视频路径
    audio_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/audio/cough/5.wav'  # 音频路径
    video_output_folder = 'video_frames'  # 视频帧输出文件夹
    audio_output_path = 'processed_audio.wav'  # 处理后的音频输出路径

    # 处理视频
    process_video(video_path, video_output_folder)

    # 处理音频
    process_audio(audio_path, audio_output_path)


if __name__ == "__main__":
    main()
