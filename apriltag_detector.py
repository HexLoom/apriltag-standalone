#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AprilTag检测主程序

这个程序从USB相机读取图像，检测AprilTag，并计算其位姿
可以作为独立程序使用，不依赖ROS

使用方法:
    python apriltag_detector.py [config_path]

参数:
    config_path: 可选的配置文件路径(默认为config/vision/tags_36h11_all.json)
"""

import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
import argparse

# 使用新的模块化结构导入
from lib.apriltag import Detector, DetectorOptions, draw_detection_results
from lib.utils.config import AprilTagConfig, ArchiveConfig, read_json, read_camera_info, matrix_to_quaternion
from lib.utils.filesystem import create_dirs_if_not_exist

def main():
    """主程序入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AprilTag检测程序')
    parser.add_argument('config_path', nargs='?', default='config/vision/tags_36h11_all.json',
                        help='配置文件路径(默认: config/vision/tags_36h11_all.json)')
    parser.add_argument('--camera', type=int, default=0,
                        help='相机设备ID(默认: 0)')
    parser.add_argument('--camera_info', default='config/camera/camera_info_1.yaml',
                        help='相机参数文件路径(默认: config/camera/camera_info_1.yaml)')
    parser.add_argument('--width', type=int, default=1280,
                        help='相机宽度(默认: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='相机高度(默认: 720)')
    args = parser.parse_args()
    
    # 读取配置
    config_path = args.config_path
    camera_info_path = args.camera_info
    
    try:
        print(f"读取配置文件: {config_path}")
        config_json = read_json(config_path)
        apriltag_config = AprilTagConfig(**config_json["AprilTagConfig"])
        archive_config = ArchiveConfig(**config_json["Archive"])
        
        print(f"读取相机参数: {camera_info_path}")
        K, D = read_camera_info(camera_info_path)
        print(f"相机内参矩阵K:\n{K}")
        print(f"畸变系数D: {D}")
    except Exception as e:
        print(f"配置读取错误: {e}")
        print("使用默认参数继续...")
        # 创建默认配置
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        D = np.zeros((4, 1), dtype=np.float32)
        
        class DefaultConfig:
            def __init__(self):
                self.family = "tag36h11"
                self.size = 0.1
                self.threads = 2
                self.max_hamming = 0
                self.z_up = True
                self.decimate = 1.0
                self.blur = 0.0
                self.refine_edges = 1
                self.debug = 0
                
        apriltag_config = DefaultConfig()
        
        class DefaultArchive:
            def __init__(self):
                self.enable = False
                self.preview = True
                self.save_raw = False
                self.save_pred = False
                self.preview_delay = 1
                self.path = "./data/apriltag"
                
        archive_config = DefaultArchive()
    
    # 初始化AprilTag检测器
    print("初始化AprilTag检测器...")
    # 确保families参数是字符串格式
    detector = Detector(DetectorOptions(
        families=apriltag_config.family,  # family已经是字符串格式
        nthreads=apriltag_config.threads,
        quad_blur=apriltag_config.blur,
        quad_decimate=apriltag_config.decimate,
        refine_edges=bool(apriltag_config.refine_edges),
        debug=bool(apriltag_config.debug)
    ))
    
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
    
    print("开始AprilTag检测循环...")
    print("按 'q' 退出")
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取相机帧")
                break
            
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测AprilTags
            start_time = time.time()
            tags = detector.detect(gray)
            end_time = time.time()
            if (end_time - start_time) != 0:
                fps = 1.0 / (end_time - start_time)
            
            # 处理检测结果
            print(f"检测到 {len(tags)} 个AprilTag, FPS: {fps:.1f}")
            
            # 为每个标签计算位姿
            for tag in tags:
                # 提取角点
                camera_apriltag_corners = np.array(tag.corners, dtype=np.float32)
                
                # 世界坐标系中的角点 (标签坐标系)
                world_apriltag_corners = np.array([
                    [-apriltag_config.size/2, apriltag_config.size/2, 0],
                    [apriltag_config.size/2, apriltag_config.size/2, 0],
                    [apriltag_config.size/2, -apriltag_config.size/2, 0],
                    [-apriltag_config.size/2, -apriltag_config.size/2, 0]
                ], dtype=np.float32)
                
                # 使用solvePnP计算位姿
                _, rvec, tvec = cv2.solvePnP(
                    world_apriltag_corners, 
                    camera_apriltag_corners, 
                    K, D,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                # 如果需要翻转Z轴
                if apriltag_config.z_up:
                    tvec[2] = -tvec[2]
                    # 旋转向量也需要调整
                    rvec[0] = -rvec[0]
                    rvec[1] = -rvec[1]
                
                # 将旋转向量转换为旋转矩阵
                R, _ = cv2.Rodrigues(rvec)
                tvec = tvec.flatten()
                quat = matrix_to_quaternion(R)
                
                # 输出位姿信息
                print(f"Tag ID: {tag.tag_id}")
                print(f"位置(米): {tvec}")
                print(f"四元数: {quat}")
            
            # 在图像上绘制检测结果
            output_image = draw_detection_results(
                frame, 
                tags, 
                K, D, 
                apriltag_config.size,
                flip_z=apriltag_config.z_up
            )
            
            # 添加FPS显示
            cv2.putText(output_image, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            if archive_config.preview:
                cv2.imshow("AprilTag", output_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # 保存图像
            if archive_config.enable:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if archive_config.save_raw:
                    raw_image_name = os.path.join(archive_config.path, f"{timestamp}_raw.png")
                    cv2.imwrite(raw_image_name, frame)
                
                if archive_config.save_pred:
                    pred_image_name = os.path.join(archive_config.path, f"{timestamp}_pred.png")
                    cv2.imwrite(pred_image_name, output_image)
    
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        print("释放资源...")
        cap.release()
        cv2.destroyAllWindows()
        print("程序结束")

if __name__ == "__main__":
    main() 