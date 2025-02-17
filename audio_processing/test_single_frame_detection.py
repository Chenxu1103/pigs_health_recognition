# pigs_health_recognition/src/video_processing/test_single_frame_detection.py

import cv2
import os
import logging
from pigs_health_recognition.src.video_processing.object_detection import detect_pigs_no_nms, load_yolov8_model

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def test_frame_detection_no_nms(yolo_model, frame, output_dir, frame_index):
    # 检测 'pig' 类别，类别ID=0，无NMS
    detected_pigs = detect_pigs_no_nms(yolo_model, frame, conf_threshold=0.2, class_ids=[0])
    logging.info(f"测试帧 {frame_index} 检测到的猪数量：{len(detected_pigs)}")

    for bbox in detected_pigs:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色框
        cv2.putText(frame, "Coughing Pig", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)  # 绿色文本

    # 保存检测结果
    test_frame_path = os.path.join(output_dir, f"test_frame_no_nms_{frame_index}.jpg")
    cv2.imwrite(test_frame_path, frame)
    logging.info(f"保存检测帧到：{test_frame_path}")


if __name__ == "__main__":
    video_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/video/202309151350（疑似）.mp4'
    output_dir = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/output'

    os.makedirs(output_dir, exist_ok=True)

    # 加载 YOLOv8 模型
    weights_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/video_processing/runs/detect/train5/weights/best.pt'  # 更新为您的训练模型路径
    try:
        yolo_model = load_yolov8_model(weights_path)
        logging.info("YOLOv8 模型已加载。")
    except Exception as e:
        logging.error("YOLOv8 模型加载失败。")
        exit(1)

    # 打印模型类别名称
    logging.info(f"YOLO类别名称: {yolo_model.names}")

    # 选择测试帧索引，例如第100帧
    frame_index = 100
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if ret:
        test_frame_detection_no_nms(yolo_model, frame, output_dir, frame_index)
    else:
        logging.error(f"无法读取帧索引 {frame_index}。")
