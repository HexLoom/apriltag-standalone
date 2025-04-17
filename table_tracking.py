#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
桌面AprilTag跟踪系统

这个程序用于跟踪桌面上的多个AprilTag标签，包括四个角落固定的参考标签(ID 0-3)
以及一个可移动的标签(ID 4)。程序会计算移动标签相对于参考标签的位置，
并使用3D可视化展示所有标签的空间关系。

使用方法:
    python table_tracking.py [--camera CAMERA_ID] [--config CONFIG_PATH]

参数:
    --camera: 相机设备ID (默认: 0)
    --config: 配置文件路径 (默认: config/vision/table_setup.json)
    --camera_info: 相机参数文件 (默认: config/camera/camera_info_1.yaml)
    --width: 相机宽度 (默认: 1280)
    --height: 相机高度 (默认: 720)
"""

import argparse
import sys
import time
import cv2
import matplotlib.pyplot as plt

# 导入自定义模块
from modules.config import load_config
from modules.utils import create_dirs_if_not_exist
from modules.visualizer import TableVisualizer
from modules.tracker import TableTracker

# 导入AprilTag检测器
from lib.detector import Detector, DetectorOptions, draw_detection_results

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='桌面AprilTag跟踪系统')
    parser.add_argument('--camera', type=int, default=0,
                       help='相机设备ID (默认: 0)')
    parser.add_argument('--config', default='config/vision/table_setup.json',
                       help='配置文件路径 (默认: config/vision/table_setup.json)')
    parser.add_argument('--camera_info', default='config/camera/camera_info_1.yaml',
                       help='相机参数文件 (默认: config/camera/camera_info_1.yaml)')
    parser.add_argument('--width', type=int, default=1280,
                       help='相机宽度分辨率 (默认: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='相机高度分辨率 (默认: 720)')
    args = parser.parse_args()

    # 加载配置
    apriltag_config, archive_config, table_config, K, D = load_config(args.config, args.camera_info)

    # 初始化可视化
    visualizer = TableVisualizer()

    # 初始化AprilTag检测器
    print("初始化AprilTag检测器...")
    detector = Detector(DetectorOptions(
        families=apriltag_config.family,
        nthreads=apriltag_config.threads,
        quad_blur=apriltag_config.blur,
        quad_decimate=apriltag_config.decimate,
        refine_edges=bool(apriltag_config.refine_edges),
        debug=bool(apriltag_config.debug)
    ))
    
    # 初始化跟踪器
    tracker = TableTracker(
        detector=detector,
        camera_matrix=K,
        dist_coeffs=D,
        apriltag_config=apriltag_config,
        table_config=table_config,
        archive_config=archive_config
    )
    
    # 初始化相机
    print(f"打开相机 ID: {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开相机 {args.camera}")
        sys.exit(1)
    
    # 设置相机分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 创建存档目录
    if archive_config.enable:
        create_dirs_if_not_exist(archive_config.path)
    
    print("开始AprilTag跟踪...")
    print("按 'q' 键退出")
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取相机帧")
                break
            
            # 处理帧 得到标签位姿，检测到的标签，已经打好标签的图像，fps(在图像中已经显示)
            tag_poses, tags, output_image, fps = tracker.process_frame(frame)
            
            # 在图像上绘制检测结果
            output_image = draw_detection_results(
                output_image, 
                tags, 
                K, D, 
                apriltag_config.size,
                flip_z=apriltag_config.z_up
            )
            
            # 更新3D可视化
            visualizer.update(tag_poses, table_config.moving_tag, table_config.reference_tags)
            
            # 显示结果
            if archive_config.preview:
                cv2.imshow("AprilTag Desktop tracking", output_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        print("释放资源...")
        cap.release()
        cv2.destroyAllWindows()
        plt.close()
        print("程序结束")

if __name__ == "__main__":
    main()