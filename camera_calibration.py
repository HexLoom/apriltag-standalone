#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
相机标定程序

使用棋盘格图案对相机进行标定，生成相机内参和畸变系数
标定结果会保存为YAML文件，可直接用于AprilTag检测

使用方法:
    python camera_calibration.py [--size WIDTH HEIGHT] [--square SQUARE_SIZE] [--output OUTPUT_FILE]

参数:
    --size: 棋盘格内角点数量，宽x高 (默认: 9x6)
    --square: 棋盘格方块大小，单位米 (默认: 0.025)
    --output: 输出文件路径 (默认: config/camera/camera_info_1.yaml)
"""

import os
import sys
import argparse
import numpy as np
import cv2
import yaml
import time
from datetime import datetime

# 使用模块化结构导入
from lib.camera import save_camera_calibration
from lib.utils.filesystem import create_dirs_if_not_exist

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='相机标定程序')
    parser.add_argument('--size', type=str, default='9x6',
                        help='棋盘格内角点数量，宽x高 (默认: 9x6)')
    parser.add_argument('--square', type=float, default=0.025,
                        help='棋盘格方块大小，单位米 (默认: 0.025)')
    parser.add_argument('--output', type=str, default='config/camera/camera_info_1.yaml',
                        help='输出文件路径 (默认: config/camera/camera_info_1.yaml)')
    parser.add_argument('--camera', type=int, default=0,
                        help='相机设备ID (默认: 0)')
    parser.add_argument('--width', type=int, default=1280,
                        help='相机宽度 (默认: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='相机高度 (默认: 720)')
    parser.add_argument('--samples', type=int, default=20,
                        help='标定样本数量 (默认: 20)')
    parser.add_argument('--preview', action='store_true',
                        help='预览未校正和校正后的图像')
    args = parser.parse_args()
    
    # 解析棋盘格尺寸
    try:
        board_w, board_h = map(int, args.size.split('x'))
    except ValueError:
        print(f"错误: 棋盘格尺寸格式不正确。应为'宽x高'，例如'9x6'")
        sys.exit(1)
    
    # 打印标定参数
    print(f"标定参数:")
    print(f"  棋盘格尺寸: {board_w}x{board_h}")
    print(f"  方格大小: {args.square} 米")
    print(f"  输出文件: {args.output}")
    print(f"  相机ID: {args.camera}")
    print(f"  图像分辨率: {args.width}x{args.height}")
    print(f"  标定样本数: {args.samples}")
    
    # 初始化相机
    print(f"打开相机 ID: {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开相机 {args.camera}")
        sys.exit(1)
    
    # 设置相机分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 获取实际相机分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"实际相机分辨率: {actual_width}x{actual_height}")
    
    # 准备标定
    # 棋盘格模式的世界坐标（Z=0）
    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2) * args.square
    
    # 存储棋盘格角点的世界坐标和图像坐标
    objpoints = []  # 3D点的世界坐标
    imgpoints = []  # 2D点的图像坐标
    
    # 计数器
    num_samples = 0
    last_detection_time = time.time()
    detection_interval = 1.0  # 两次检测之间的最小时间间隔（秒）
    
    print("\n开始标定过程...")
    print("请将棋盘格放在相机前，从不同角度拍摄。")
    print("保持棋盘格在视野内，程序会自动检测并收集样本。")
    print("按 'q' 退出，按 's' 手动保存当前帧作为样本")
    
    while num_samples < args.samples:
        # 读取相机帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取相机帧")
            break
        
        # 显示提示信息
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Samples: {num_samples}/{args.samples}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to save, 'q' to quit", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        
        # 自动模式：每隔一段时间尝试检测
        current_time = time.time()
        auto_detect = (current_time - last_detection_time) > detection_interval
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 寻找棋盘格角点
        if auto_detect:
            found, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)
            
            # 如果找到角点
            if found:
                # 亚像素精确化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 绘制角点
                cv2.drawChessboardCorners(display_frame, (board_w, board_h), corners2, found)
                
                # 是否保存当前样本
                save_sample = True
                
                # 检查与已有样本的差异性
                if imgpoints and len(imgpoints) > 0:
                    # 计算与上一个样本的平均距离
                    last_corners = imgpoints[-1].reshape(-1, 2)
                    curr_corners = corners2.reshape(-1, 2)
                    mean_dist = np.mean(np.linalg.norm(last_corners - curr_corners, axis=1))
                    
                    # 如果差异太小，忽略这个样本
                    if mean_dist < 10.0:  # 像素距离阈值
                        save_sample = False
                
                if save_sample:
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    num_samples += 1
                    last_detection_time = current_time
                    print(f"自动采集样本 {num_samples}/{args.samples}")
                    
                    # 增加视觉反馈
                    cv2.putText(display_frame, "样本已采集!", (actual_width//2-100, actual_height//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # 显示图像
        cv2.imshow('相机标定', display_frame)
        key = cv2.waitKey(1)
        
        # 按's'键手动保存样本
        if key == ord('s'):
            # 再次查找角点（确保最新状态）
            found, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)
            
            if found:
                # 亚像素精确化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 保存样本
                objpoints.append(objp)
                imgpoints.append(corners2)
                num_samples += 1
                last_detection_time = current_time
                print(f"手动采集样本 {num_samples}/{args.samples}")
                
                # 显示保存的样本
                sample_display = frame.copy()
                cv2.drawChessboardCorners(sample_display, (board_w, board_h), corners2, found)
                cv2.putText(sample_display, f"样本 {num_samples} 已保存", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('采集的样本', sample_display)
                cv2.waitKey(500)  # 显示半秒
            else:
                print("未检测到棋盘格，无法保存样本")
        
        # 按'q'键退出
        if key == ord('q'):
            break
    
    # 释放相机
    cap.release()
    cv2.destroyAllWindows()
    
    # 检查是否有足够的样本
    if num_samples < 3:
        print(f"错误: 采集的样本数量不足，需要至少3个样本，实际只有 {num_samples} 个")
        sys.exit(1)
        
    print(f"\n开始标定计算...")
    print(f"使用 {num_samples} 个样本进行标定")
    
    # 执行相机标定
    ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (actual_width, actual_height), None, None)
    
    # 输出标定结果
    print("\n标定结果:")
    print(f"重投影误差: {ret}")
    print(f"相机内参矩阵:\n{camera_matrix}")
    print(f"畸变系数: {dist_coefs.ravel()}")
    
    # 保存标定结果
    save_camera_calibration(args.output, camera_matrix, dist_coefs, actual_width, actual_height)
    
    # 如果需要预览校正效果
    if args.preview and num_samples > 0:
        print("\n预览校正效果...")
        # 重新打开相机
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("无法重新打开相机进行预览")
            sys.exit(0)
        
        # 重新设置相机分辨率以确保预览帧尺寸正确
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        # (可选) 再次检查实际分辨率
        preview_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        preview_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if preview_height != actual_height or preview_width != actual_width:
            print(f"警告: 预览时的相机分辨率 ({preview_width}x{preview_height}) 与标定时 ({actual_width}x{actual_height}) 不一致，预览可能不准确。")

        # 计算校正映射
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coefs, None, camera_matrix, 
            (actual_width, actual_height), cv2.CV_32FC1)
        
        print("按'q'退出预览")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 未校正的图像
            cv2.putText(frame, "Original", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 校正后的图像
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            cv2.putText(undistorted, "Undistorted", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 并排显示
            combined = np.hstack((frame, undistorted))
            cv2.imshow('校正效果对比', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    print("相机标定完成")
    
if __name__ == "__main__":
    main() 