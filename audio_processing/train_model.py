from model import train_cough_classifier, get_audio_files_from_dir

def main():
    # 假设你已经有一个包含咳嗽和非咳嗽音频文件的目录路径
    cough_audio_dir = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/audio/cough'
    non_cough_audio_dir = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/audio/non_cough'

    # 获取所有音频文件路径
    cough_files = get_audio_files_from_dir(cough_audio_dir)
    non_cough_files = get_audio_files_from_dir(non_cough_audio_dir)

    # 训练咳嗽分类器
    model = train_cough_classifier(cough_files, non_cough_files)

    # 保存模型
    from joblib import dump
    dump(model, 'cough_classifier_model.joblib1')

    print("模型已保存。")

if __name__ == "__main__":
    main()
