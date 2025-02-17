# audio_visualization.py

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import logging
from scipy.io import wavfile
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 设置日志配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_audio_from_video(video_path, audio_path):
    """
    使用FFmpeg从视频中提取音频。

    :param video_path: 视频文件路径
    :param audio_path: 输出音频文件路径
    """
    import subprocess
    command = [
        'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '44100', '-ac', '2', audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.info("音频提取成功。")
    except subprocess.CalledProcessError:
        logging.error("音频提取失败。请确保FFmpeg已正确安装并配置。")
        raise


def predict_cough_events(audio_path, model, scaler, threshold=0.5):
    """
    使用训练好的模型预测音频中的咳嗽事件。

    :param audio_path: 音频文件路径
    :param model: 训练好的咳嗽检测模型
    :param scaler: 标准化器，用于预处理音频特征
    :param threshold: 预测阈值，超过此值的为咳嗽事件
    :return: 检测到的咳嗽事件时间戳列表（秒）
    """
    # 加载音频
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 提取Delta特征（一阶差分）
    delta_mfcc = librosa.feature.delta(mfcc)

    # 提取其他音频特征
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # (1, n_frames)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)  # (1, n_frames)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # (1, n_frames)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # (1, n_frames)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # (n_bands, n_frames)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)  # (1, n_frames)
    rms = librosa.feature.rms(y=y)  # (1, n_frames)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # (12, n_frames)

    # 计算每个特征的均值，确保总共8个其他特征
    spectral_centroid_mean = np.mean(spectral_centroid)  # 1
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)  # 1
    spectral_rolloff_mean = np.mean(spectral_rolloff)  # 1
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)  # 1
    spectral_contrast_mean = np.mean(spectral_contrast)  # 1
    spectral_flatness_mean = np.mean(spectral_flatness)  # 1
    rms_mean = np.mean(rms)  # 1
    chroma_mean = np.mean(chroma)  # 1

    # 其他8个特征的均值
    other_features_mean = np.array([
        spectral_centroid_mean,
        zero_crossing_rate_mean,
        spectral_rolloff_mean,
        spectral_bandwidth_mean,
        spectral_contrast_mean,
        spectral_flatness_mean,
        rms_mean,
        chroma_mean
    ])  # (8,)

    # 获取帧数
    n_frames = mfcc.shape[1]

    # 重复其他特征均值以匹配帧数
    other_features_repeated = np.tile(other_features_mean, (n_frames, 1))  # (n_frames, 8)

    # 转置MFCC和Delta MFCC特征并拼接所有特征
    final_features = np.hstack([
        mfcc.T,  # (n_frames, 13)
        delta_mfcc.T,  # (n_frames, 13)
        other_features_repeated  # (n_frames, 8)
    ])  # 总共 13 + 13 + 8 = 34

    # 打印特征形状以调试
    logging.debug(f"Final features shape: {final_features.shape}")  # 应为 (n_frames, 34)

    # 标准化特征
    try:
        features_scaled = scaler.transform(final_features)  # (n_frames, 34)
    except ValueError as ve:
        logging.error(f"特征标准化失败: {ve}")
        raise

    # 模型预测
    if hasattr(model, 'predict_proba'):
        predictions = model.predict_proba(features_scaled)[:, 1]  # 获取正类概率
    else:
        # 如果模型没有 predict_proba 方法，例如某些支持向量机
        predictions = model.decision_function(features_scaled)
        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())  # 归一化为0-1

    # 获取预测概率超过阈值的帧索引
    cough_indices = np.where(predictions >= threshold)[0]

    # 转换为时间戳
    cough_timestamps = librosa.frames_to_time(cough_indices, sr=sr, hop_length=512)

    # 去除近邻的重复事件（假设咳嗽事件至少间隔0.5秒）
    if len(cough_timestamps) == 0:
        return []

    filtered_timestamps = [cough_timestamps[0]]
    for ts in cough_timestamps[1:]:
        if ts - filtered_timestamps[-1] > 0.5:
            filtered_timestamps.append(ts)

    return filtered_timestamps


def visualize_audio(audio_path, cough_timestamps, output_dir):
    """
    可视化音频波形及咳嗽事件，并生成频谱图。

    :param audio_path: 音频文件路径
    :param cough_timestamps: 检测到的咳嗽事件时间戳列表（秒）
    :param output_dir: 输出目录路径
    """
    try:
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        time = np.linspace(0., duration, len(y))

        # 创建一个包含两个子图的图像：波形图和频谱图
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))

        # 波形图
        axs[0].plot(time, y, label='Audio Waveform')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Audio Waveform with Cough Events')

        # 标记咳嗽事件
        for ts in cough_timestamps:
            axs[0].axvline(x=ts, color='r', linestyle='--', label='Cough' if ts == cough_timestamps[0] else "")

        axs[0].legend(loc='upper right')

        # 频谱图
        n_fft = 1024
        hop_length = 512
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=axs[1])
        axs[1].set_title('Spectrogram')
        fig.colorbar(img, ax=axs[1], format="%+2.f dB")

        # 标记咳嗽事件
        for ts in cough_timestamps:
            axs[1].axvline(x=ts, color='r', linestyle='--', label='Cough' if ts == cough_timestamps[0] else "")

        # 防止重复标签
        handles, labels = axs[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[1].legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'audio_waveform_and_spectrogram.png'), dpi=300)
        plt.close()
        logging.info("音频波形和频谱图已保存。")

    except Exception as e:
        logging.error(f"音频可视化失败: {e}")


def evaluate_model(true_labels, predicted_probs, output_dir):
    """
    评估模型性能，并生成ROC和Precision-Recall曲线。

    :param true_labels: 真实标签（1表示咳嗽，0表示非咳嗽）
    :param predicted_probs: 模型预测的概率值
    :param output_dir: 输出目录路径
    """
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(12, 5))

        # ROC曲线
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Precision-Recall曲线
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_evaluation_curves.png'), dpi=300)
        plt.close()
        logging.info("模型评估曲线已保存。")

    except Exception as e:
        logging.error(f"模型评估失败: {e}")


def main():
    """
    主函数，执行音频提取、咳嗽事件检测和可视化。
    """
    # 输入视频文件路径
    video_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/video/202309151632.mp4'  # 替换为您的视频文件路径

    # 输出目录
    output_dir = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/audio_processing'  # 替换为您的输出目录路径
    os.makedirs(output_dir, exist_ok=True)

    # 提取音频
    audio_path = os.path.join(output_dir, "audio.wav")
    extract_audio_from_video(video_path, audio_path)

    # 加载训练好的模型和标准化器
    # 此部分需要根据您的模型和标准化器的具体实现进行调整
    # 假设模型是一个已经训练好的sklearn模型，并且scaler是一个sklearn的Scaler对象
    import joblib  # 用于加载模型和Scaler

    model_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/audio_processing/cough_classifier_model.joblib'  # 替换为您的模型文件路径
    scaler_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/audio_processing/scaler.joblib'  # 替换为您的Scaler文件路径

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logging.info("模型和Scaler加载成功。")
    except Exception as e:
        logging.error(f"加载模型或Scaler失败: {e}")
        return

    # 预测咳嗽事件
    cough_timestamps = predict_cough_events(audio_path, model, scaler, threshold=0.5)
    logging.info(f"检测到的咳嗽时间戳: {cough_timestamps}")

    # 可视化检测结果
    visualize_audio(audio_path, cough_timestamps, output_dir)

    # 如果有真实标签，进行模型评估
    # 这里假设您有一个真实标签列表true_labels和模型预测的概率列表predicted_probs
    # 您需要根据实际情况加载或生成这些数据
    # 示例：
    # true_labels = [0, 1, 0, 1, ...]  # 真实标签
    # predicted_probs = [0.1, 0.8, 0.2, 0.9, ...]  # 模型预测的概率
    # evaluate_model(true_labels, predicted_probs, output_dir)


if __name__ == "__main__":
    main()
