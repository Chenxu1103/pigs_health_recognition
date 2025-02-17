# 文件路径示例：pigs_health_recognition/src/audio_processing/model.py

import librosa
import numpy as np
from joblib import load
import logging
import os

def load_model():
    """
    加载音频模型和标准化器
    """
    model_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/audio_processing/cough_classifier_model.joblib'
    scaler_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/audio_processing/scaler.joblib'

    # 检查模型文件是否存在
    if not os.path.isfile(model_path):
        logging.error(f"模型文件未找到: {model_path}")
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    # 检查标准化器文件是否存在
    if not os.path.isfile(scaler_path):
        logging.error(f"标准化器文件未找到: {scaler_path}")
        raise FileNotFoundError(f"标准化器文件未找到: {scaler_path}")

    try:
        model = load(model_path)
        logging.info(f"模型已加载: {model_path}")
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        raise e

    try:
        scaler = load(scaler_path)
        logging.info(f"标准化器已加载: {scaler_path}")
    except Exception as e:
        logging.error(f"加载标准化器失败: {e}")
        raise e

    return model, scaler

def predict_cough(audio_file, model, scaler):
    # 前置检查
    try:
        y, sr = librosa.load(audio_file, sr=44100)
        logging.info(f"音频文件加载成功: {audio_file}, 采样率: {sr}, 样本数: {len(y)}")
    except Exception as e:
        logging.error(f"加载音频文件失败: {audio_file}, 错误: {e}")
        return []

    if len(y) == 0:
        logging.warning("警告：音频信号长度为0。")
        return []

    # 音频预处理：去噪和标准化
    y = librosa.effects.preemphasis(y)
    y = librosa.util.normalize(y)

    # 动态调整帧长和n_fft参数
    min_frame_length = 1024
    desired_n_fft = 1024
    frame_length = max(min_frame_length, desired_n_fft)
    hop_length = desired_n_fft  # 保证每个窗口只有一个时间步

    # 检查音频长度是否足够
    if len(y) < frame_length:
        logging.warning(f"音频长度不足 ({len(y)} < {frame_length})，将pad音频信号。")
        y = np.pad(y, (0, frame_length - len(y)), mode='constant')

    # 分帧
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T

    if len(frames) == 0:
        logging.warning("警告：音频信号无法分帧。")
        return []

    # 提取MFCC特征
    mfcc_features = []
    for i, frame in enumerate(frames):
        try:
            mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=34, n_fft=desired_n_fft, hop_length=hop_length)
            mfcc_mean = np.mean(mfcc, axis=1)  # 单帧34维
            mfcc_features.append(mfcc_mean)
        except Exception as e:
            logging.error(f"帧 {i} MFCC特征提取失败: {e}")

    if len(mfcc_features) == 0:
        logging.warning("警告：所有帧的MFCC特征提取失败。")
        return []

    mfcc_features = np.array(mfcc_features)  # shape: (num_frames, 34)

    # 检查特征数量是否匹配
    expected_features = scaler.mean_.shape[0]
    if mfcc_features.shape[1] != expected_features:
        logging.error(f"特征数量不匹配。MFCC特征维度: {mfcc_features.shape[1]}, 标准化器期望维度: {expected_features}")
        return []

    # 特征标准化
    try:
        mfcc_scaled = scaler.transform(mfcc_features)
    except ValueError as e:
        logging.error(f"特征缩放失败：{e}")
        return []

    # 模型预测
    try:
        predictions = model.predict(mfcc_scaled)
    except Exception as e:
        logging.error(f"模型预测失败：{e}")
        return []

    # 假设预测结果为二分类，1表示咳嗽
    cough_indices = np.where(predictions == 1)[0]
    logging.info(f"检测到的咳嗽帧数: {len(cough_indices)}")

    if len(cough_indices) == 0:
        logging.info("未检测到任何咳嗽事件。")
        return []

    # 将帧索引转换为时间戳
    try:
        cough_timestamps = librosa.frames_to_time(cough_indices, sr=sr, hop_length=hop_length)
    except Exception as e:
        logging.error(f"转换帧索引到时间戳失败：{e}")
        return []

    # 去除相近的重复咳嗽事件
    filtered_timestamps = []
    prev = -0.5
    for ts in cough_timestamps:
        if ts - prev >= 0.5:
            filtered_timestamps.append(ts)
            prev = ts

    logging.info(f"过滤后的咳嗽事件数量: {len(filtered_timestamps)}")
    return filtered_timestamps
