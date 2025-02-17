import os
import shutil


def move_files(labels_src_dir, images_src_dir, labels_dest_dir, images_dest_dir, num_files):
    """
    移动指定数量的带标签文件及对应图像文件从训练集到验证集。

    :param labels_src_dir: 源标签目录（训练集标签目录）
    :param images_src_dir: 源图像目录（训练集图像目录）
    :param labels_dest_dir: 目标标签目录（验证集标签目录）
    :param images_dest_dir: 目标图像目录（验证集图像目录）
    :param num_files: 要移动的文件数量
    """
    # 获取所有带标签的文件列表
    all_labels = [f for f in os.listdir(labels_src_dir) if f.endswith('.txt')]

    if not all_labels:
        print(f"源标签目录中没有标签文件: {labels_src_dir}")
        return

    # 选择要移动的文件
    files_to_move = all_labels[:num_files]

    for label_file in files_to_move:
        label_src_path = os.path.join(labels_src_dir, label_file)
        label_dest_path = os.path.join(labels_dest_dir, label_file)

        # 移动标签文件
        shutil.move(label_src_path, label_dest_path)
        print(f"Moved label file: {label_src_path} -> {label_dest_path}")

        # 移动对应的图像文件
        base_name = os.path.splitext(label_file)[0]
        # 支持常见的图像格式
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            image_src_path = os.path.join(images_src_dir, base_name + ext)
            image_dest_path = os.path.join(images_dest_dir, base_name + ext)
            if os.path.exists(image_src_path):
                shutil.move(image_src_path, image_dest_path)
                print(f"Moved image file: {image_src_path} -> {image_dest_path}")
                break
        else:
            print(f"对应的图像文件未找到: {base_name}.[jpg|jpeg|png|bmp|gif]")


def main():
    # 定义路径（请根据实际情况修改）
    dataset_dir = "/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/dataset"
    labels_train_dir = os.path.join(dataset_dir, "labels/train_labels")
    images_train_dir = os.path.join(dataset_dir, "images/train_images")
    labels_val_dir = os.path.join(dataset_dir, "labels/val_labels")
    images_val_dir = os.path.join(dataset_dir, "images/val_images")

    # 定义要移动的文件数量（根据数据集大小调整）
    num_files_to_move = 10  # 示例：移动10个文件

    # 确保目标目录存在
    os.makedirs(labels_val_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)

    # 移动文件
    move_files(labels_train_dir, images_train_dir, labels_val_dir, images_val_dir, num_files_to_move)


if __name__ == "__main__":
    main()
