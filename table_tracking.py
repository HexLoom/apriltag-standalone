#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
桌面AprilTag跟踪系统

这个程序用于跟踪桌面上的多个AprilTag标签，包括四个角落固定的参考标签(ID 0-3)
以及多个可移动的标签(默认ID 4,5,6等)。程序支持拍照初始化，即使某些标签被遮挡，
也能根据已知标签的空间关系估计被遮挡标签的位置。

使用方法:
    python table_tracking.py [--config CONFIG_PATH]

参数:
    --config: 配置文件路径 (默认: config/vision/table_setup.json)
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
    parser.add_argument('--config', default='config/vision/table_setup.json',
                       help='配置文件路径 (默认: config/vision/table_setup.json)')
    args = parser.parse_args()

    # 加载配置
    apriltag_config, camera_config, archive_config, table_config, K, D = load_config(args.config)
    
    # 输出配置信息
    print(f"参考标签ID: {table_config.reference_tags}")
    print(f"移动标签ID: {table_config.moving_tags}")
    print(f"系统是否需要初始化: {table_config.require_initialization}")
    print(f"相机ID: {camera_config.device_id}")
    print(f"相机分辨率: {camera_config.width}x{camera_config.height}")
    
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
    print(f"打开相机 ID: {camera_config.device_id}...")
    cap = cv2.VideoCapture(camera_config.device_id)
    if not cap.isOpened():
        print(f"无法打开相机 {camera_config.device_id}")
        sys.exit(1)
    
    # 设置相机分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
    
    # 创建存档目录
    if archive_config.enable:
        create_dirs_if_not_exist(archive_config.path)
    
    print("开始AprilTag跟踪...")
    print("按 'q' 键退出，按 'i' 键手动初始化系统")
    
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
            visualizer.update(tag_poses, table_config.moving_tags, table_config.reference_tags)
            
            # 显示结果
            if archive_config.preview:
                cv2.imshow("AprilTag Desktop tracking", output_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('i'):
                    print("手动触发系统初始化...")
                    # 等待1秒确保相机稳定
                    time.sleep(1)
                    # 读取新的一帧并初始化
                    ret, init_frame = cap.read()
                    if ret:
                        success = tracker.initialize_system(init_frame)
                        if success:
                            print("手动初始化成功！")
                        else:
                            print("手动初始化失败，请确保所有标签可见。")
    
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