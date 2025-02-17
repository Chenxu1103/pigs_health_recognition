import os
import sys

def remove_empty_labels(labels_dir, images_dir):
    """
    删除 labels_dir 中的空标签文件及对应的图像文件。

    :param labels_dir: 标签文件所在目录
    :param images_dir: 图像文件所在目录
    """
    if not os.path.exists(labels_dir):
        print(f"标签目录不存在: {labels_dir}")
        return

    if not os.path.exists(images_dir):
        print(f"图像目录不存在: {images_dir}")
        return

    empty_labels = []
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        if os.path.isfile(label_path):
            if os.path.getsize(label_path) == 0:
                empty_labels.append(label_file)

    if not empty_labels:
        print(f"No empty label files found in {labels_dir}.")
        return

    print(f"Found {len(empty_labels)} empty label files in {labels_dir}.")

    for label_file in empty_labels:
        label_path = os.path.join(labels_dir, label_file)
        base_name = os.path.splitext(label_file)[0]
        # 支持常见的图像格式
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            image_file = base_name + ext
            image_path = os.path.join(images_dir, image_file)
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print(f"Deleted image file: {image_path}")
                except Exception as e:
                    print(f"Error deleting image file {image_path}: {e}")
            else:
                print(f"Corresponding image file not found: {image_path}")
        # 删除空的标签文件
        try:
            os.remove(label_path)
            print(f"Deleted empty label file: {label_path}")
        except Exception as e:
            print(f"Error deleting label file {label_path}: {e}")

def main():
    # 定义路径（请根据实际情况修改）
    dataset_dir = "/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/dataset"
    labels_subdirs = ["labels/train_labels", "labels/val_labels"]
    images_subdirs = ["images/train_images", "images/val_images"]

    for labels_subdir, images_subdir in zip(labels_subdirs, images_subdirs):
        labels_dir = os.path.join(dataset_dir, labels_subdir)
        images_dir = os.path.join(dataset_dir, images_subdir)
        print(f"\nProcessing labels directory: {labels_dir}")
        remove_empty_labels(labels_dir, images_dir)

if __name__ == "__main__":
    main()
