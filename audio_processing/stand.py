import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# 定义音频文件路径
COUGH_AUDIO_DIR = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/audio/cough'
NON_COUGH_AUDIO_DIR = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/audio/non_cough'

# 提取音频特征的函数
# 提取音频特征的函数
def extract_features(audio_path):
    """从音频信号中提取特征"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0:
            print(f"警告：音频文件 {audio_path} 是空的！")
            return None

        # 提取 MFCC 特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        # 提取 Chroma 特征
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # 提取 Spectral Contrast 特征
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bands=6)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        # 将特征合并成一个大数组
        features = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean])

        # 确保特征数量为 34
        if features.shape[0] < 34:
            features = np.pad(features, (0, 34 - features.shape[0]), 'constant', constant_values=0)
        elif features.shape[0] > 34:
            features = features[:34]

        return features

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# 提取所有音频文件的特征和标签
def extract_all_features(cough_dir, non_cough_dir):
    features = []
    labels = []

    # 处理咳嗽音频
    for file_name in os.listdir(cough_dir):
        if file_name.endswith('.wav'):
            audio_path = os.path.join(cough_dir, file_name)
            feature = extract_features(audio_path)
            if feature is not None:
                features.append(feature)
                labels.append(1)  # 咳嗽的标签是 1

    # 处理非咳嗽音频
    for file_name in os.listdir(non_cough_dir):
        if file_name.endswith('.wav'):
            audio_path = os.path.join(non_cough_dir, file_name)
            feature = extract_features(audio_path)
            if feature is not None:
                features.append(feature)
                labels.append(0)  # 非咳嗽的标签是 0

    return np.array(features), np.array(labels)
# 提取所有音频文件的特征和标签
# 提取特征和标签
features, labels = extract_all_features(COUGH_AUDIO_DIR, NON_COUGH_AUDIO_DIR)

# 检查提取的特征是否为空
if features.size == 0:
    print("没有提取到有效的特征，检查音频文件或特征提取过程。")
else:
    # 打印特征的形状以帮助调试
    print(f"提取的特征形状: {features.shape}")

    # 检查是否有多维数据
    if len(features.shape) == 1:
        features = features.reshape(1, -1)  # 转换为二维数组

    # 确保每个特征向量的维度为 34
    if features.shape[1] != 34:
        print(f"特征的维度不正确: {features.shape[1]}，期望的是 34。")
    else:
        # 创建标准化器
        scaler = StandardScaler()

        # 用所有音频特征拟合标准化器
        features_scaled = scaler.fit_transform(features)

        # 保存标准化器
        scaler_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/audio_processing/scaler.joblib'
        joblib.dump(scaler, scaler_path)

        print(f"StandardScaler 已保存到 {scaler_path}")

        # 使用支持向量机（SVM）训练分类器
        model = SVC(kernel='linear', probability=True)  # 线性核支持向量机

        # 训练模型
        model.fit(features_scaled, labels)

        # 保存训练好的分类器
        model_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/audio_processing/cough_classifier_model.joblib'
        joblib.dump(model, model_path)

        print(f"训练好的分类器已保存到 {model_path}")
