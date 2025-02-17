import os
import json
import shutil
import random
import cv2


def convert_json_to_yolo(json_path, source_images_dir, target_images_dir, target_labels_dir):
    """
    将单个 JSON 文件转换为 YOLOv8 格式的标签文件，并复制图像到目标目录。

    :param json_path: JSON 注释文件路径
    :param source_images_dir: 原始图像目录
    :param target_images_dir: 目标图像目录（训练或验证）
    :param target_labels_dir: 目标标签目录（训练或验证）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_filename = data.get('imagePath')
    if not image_filename:
        print(f"JSON 文件 {json_path} 中缺少 'imagePath' 键。")
        return

    source_image_path = os.path.join(source_images_dir, image_filename)
    if not os.path.exists(source_image_path):
        print(f"图像文件不存在: {source_image_path}")
        return

    # 读取图像以获取尺寸
    image = cv2.imread(source_image_path)
    if image is None:
        print(f"无法读取图像文件: {source_image_path}")
        return

    img_height, img_width, _ = image.shape

    # 复制图像到目标目录
    shutil.copy(source_image_path, target_images_dir)

    # 创建标签文件
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_filepath = os.path.join(target_labels_dir, label_filename)

    with open(label_filepath, 'w', encoding='utf-8') as lf:
        for shape in data.get('shape', []):
            label = shape.get('label')
            boxes = shape.get('boxes')

            if label is None or boxes is None:
                continue  # 跳过缺少信息的标注

            if label.lower() != 'pig':
                continue  # 只处理 'pig' 类别

            if len(boxes) != 4:
                print(f"标注框格式错误: {boxes} in {json_path}")
                continue  # 跳过格式错误的标注

            x1, y1, x2, y2 = boxes
            # 计算 YOLO 格式的归一化坐标
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            class_id = 0  # 假设 'pig' 的类别 ID 为 0

            # 写入标签文件
            lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def prepare_dataset(source_images_dir, source_annotations_dir, dataset_dir, train_split=0.8):
    """
    准备数据集，转换标注并组织训练集和验证集目录。

    :param source_images_dir: 原始图像目录
    :param source_annotations_dir: 原始 JSON 注释目录
    :param dataset_dir: 数据集根目录
    :param train_split: 训练集占比
    """
    # 定义目标目录
    train_images_dir = os.path.join(dataset_dir, 'images', 'train_images')
    val_images_dir = os.path.join(dataset_dir, 'images', 'val_images')
    train_labels_dir = os.path.join(dataset_dir, 'labels', 'train_labels')
    val_labels_dir = os.path.join(dataset_dir, 'labels', 'val_labels')

    # 创建目标目录（如果不存在）
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 获取所有 JSON 注释文件
    all_json_files = [f for f in os.listdir(source_annotations_dir) if f.endswith('.json')]

    # 打乱顺序
    random.seed(42)
    random.shuffle(all_json_files)

    # 划分训练集和验证集
    train_size = int(len(all_json_files) * train_split)
    train_json_files = all_json_files[:train_size]
    val_json_files = all_json_files[train_size:]

    print(f"总数据量: {len(all_json_files)}")
    print(f"训练集: {len(train_json_files)}")
    print(f"验证集: {len(val_json_files)}")

    # 处理训练集
    print("正在处理训练集...")
    for json_file in train_json_files:
        json_path = os.path.join(source_annotations_dir, json_file)
        convert_json_to_yolo(json_path, source_images_dir, train_images_dir, train_labels_dir)

    # 处理验证集
    print("正在处理验证集...")
    for json_file in val_json_files:
        json_path = os.path.join(source_annotations_dir, json_file)
        convert_json_to_yolo(json_path, source_images_dir, val_images_dir, val_labels_dir)

    print("数据集准备完成！")


if __name__ == "__main__":
    # 定义路径（请根据实际情况修改）
    source_images_dir = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/dataset/猪只数据集/train_img'
    source_annotations_dir = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/dataset/猪只数据集/train_json'
    dataset_dir = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/dataset'

    prepare_dataset(source_images_dir, source_annotations_dir, dataset_dir, train_split=0.8)
