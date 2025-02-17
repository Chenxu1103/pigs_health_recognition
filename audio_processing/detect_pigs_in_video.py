# pigs_health_recognition/src/video_processing/annotate_coughing_pigs.py

from ultralytics import YOLO
import cv2
import os
import logging

# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def annotate_coughing_pigs(model_path, input_video_path, output_video_path, cough_timestamps, fps, conf_threshold=0.2):
    """
    在视频中标注咳嗽的猪。

    :param model_path: 模型权重文件路径
    :param input_video_path: 输入视频文件路径
    :param output_video_path: 输出视频文件路径
    :param cough_timestamps: 咳嗽事件的时间戳列表（秒）
    :param fps: 视频帧率
    :param conf_threshold: 置信度阈值
    """
    # 加载模型
    model = YOLO(model_path)
    logging.info("YOLOv8模型已加载。")

    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"无法打开视频文件: {input_video_path}")
        return

    # 获取视频属性
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用适当的编码器

    # 创建视频写入对象
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (width, height))
    logging.info(f"开始处理视频，输出路径: {output_video_path}")

    # 将时间戳转换为帧编号
    cough_frames = [int(ts * fps) for ts in cough_timestamps]
    logging.info(f"咳嗽事件对应的帧编号: {cough_frames}")

    frame_number = 0
    detected_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        logging.debug(f"处理帧数: {frame_number}")

        # 使用模型进行预测
        results = model.predict(source=frame, conf=conf_threshold, verbose=False)

        # 获取检测结果
        detections = results[0].boxes
        num_detections = len(detections)
        detected_count += num_detections
        logging.debug(f"帧 {frame_number} 检测到的猪数量: {num_detections}")

        # 检查当前帧是否有咳嗽事件
        if frame_number in cough_frames:
            logging.debug(f"帧 {frame_number} 有咳嗽事件")
            # 标注“coughing_pig”
            for box in detections:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[cls_id]
                if label == 'coughing_pig':
                    # 在框上添加文本
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 可视化结果
        annotated_frame = results[0].plot()

        # 写入输出视频
        out.write(annotated_frame)

    # 释放资源
    cap.release()
    out.release()
    logging.info(f"视频处理完成，总处理帧数: {frame_number}, 总检测到的猪数量: {detected_count}")


if __name__ == "__main__":
    model_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/src/video_processing/output/runs/detect/train5/weights/best.pt'  # 更新为您的模型路径
    input_video_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/data/video/202309151308（疑似）.mp4'  # 更新为您的视频路径
    output_video_path = '/Users/Chenxu/养殖猪健康状况检测项目/pigs_health_recognition/output/annotated_video.mp4'  # 输出视频路径

    # 示例咳嗽事件时间戳（秒）
    cough_timestamps = [10.5, 25.3, 40.7]  # 需要根据音频检测结果填充
    fps = 30  # 视频帧率，确保与实际视频帧率一致

    annotate_coughing_pigs(model_path, input_video_path, output_video_path, cough_timestamps, fps, conf_threshold=0.2)
