#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
相机参数验证工具

用于验证相机标定参数是否准确，通过以下方式：
1. 棋盘格图像校正测试
2. 距离测量测试
3. 坐标系可视化

使用方法:
    python verify_camera.py [--camera_info CAMERA_INFO_PATH]
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time

from lib.utils.config import read_camera_info
from lib.utils.filesystem import create_dirs_if_not_exist
from lib.apriltag import Detector, DetectorOptions, draw_detection_results

def test_undistortion(K, D, img=None):
    """测试图像校正
    
    参数:
        K: 相机内参矩阵
        D: 畸变系数
        img: 输入图像，如果为None则从相机获取
    """
    # 如果没有输入图像，从相机获取
    if img is None:
        print("从相机获取图像...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开相机")
            return
        
        print("按's'保存当前帧进行校正测试，按'q'退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取相机帧")
                break
                
            cv2.imshow("Camera Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                img = frame.copy()
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
    
    # 校正图像
    h, w = img.shape[:2]
    
    # 获取最优相机矩阵
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    
    # 使用undistort方法
    dst1 = cv2.undistort(img, K, D, None, newcameramtx)
    
    # 使用重映射方法
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (w, h), cv2.CV_32FC1)
    dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    
    # 裁剪感兴趣区域
    x, y, w, h = roi
    if all(v > 0 for v in [x, y, w, h]):
        dst1_roi = dst1[y:y+h, x:x+w]
        dst2_roi = dst2[y:y+h, x:x+w]
    else:
        dst1_roi = dst1
        dst2_roi = dst2
    
    # 调整图像大小，以便于比较
    img_resized = cv2.resize(img, (800, 600))
    dst1_resized = cv2.resize(dst1, (800, 600))
    dst2_resized = cv2.resize(dst2, (800, 600))
    
    # 创建比较视图
    comparison = np.hstack((img_resized, dst1_resized, dst2_resized))
    
    # 添加标签
    label_height = 40
    background = np.zeros((label_height, comparison.shape[1], 3), dtype=np.uint8)
    cv2.putText(background, "Original Image", (400-60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(background, "Undistort Method", (1200-90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(background, "Remap Method", (2000-60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 合并标签和图像
    result = np.vstack((background, comparison))
    
    # 显示结果
    cv2.imshow("Image Undistortion Test", result)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return dst1

def test_distance_measurement(K, D, tag_size=0.1):
    """测试距离测量精度
    
    参数:
        K: 相机内参矩阵
        D: 畸变系数
        tag_size: AprilTag标签大小(米)
    """
    # 创建检测器
    detector = Detector(DetectorOptions(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_blur=0.0,
        refine_edges=True
    ))
    
    # 打开相机
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开相机")
        return
    
    print("\n=== AprilTag距离测量测试 ===")
    print("1. 准备一个尺寸已知的AprilTag标签")
    print(f"2. 标签尺寸设置为 {tag_size} 米")
    print("3. 使用尺子或卷尺测量标签到相机的准确距离")
    print("4. 将标签放在不同的已知距离处，检查测量结果")
    print("\n按's'记录测量结果，按'q'退出测试")
    
    measurements = []
    
    while True:
        # 读取相机帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取相机帧")
            break
        
        # 检测AprilTag
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        
        if tags:
            # 绘制检测结果
            output_image = draw_detection_results(
                frame, 
                tags, 
                K, D, 
                tag_size,
                flip_z=True
            )
            
            # 计算每个标签的距离
            for tag in tags:
                # 提取角点
                camera_corners = np.array(tag.corners, dtype=np.float32)
                
                # 世界坐标系中的角点
                world_corners = np.array([
                    [-tag_size/2, tag_size/2, 0],
                    [tag_size/2, tag_size/2, 0],
                    [tag_size/2, -tag_size/2, 0],
                    [-tag_size/2, -tag_size/2, 0]
                ], dtype=np.float32)
                
                # 计算位姿
                _, rvec, tvec = cv2.solvePnP(
                    world_corners, 
                    camera_corners, 
                    K, D,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                # 计算距离
                distance = np.linalg.norm(tvec)
                
                # 在图像上显示测量结果
                cv2.putText(output_image, f"Distance: {distance:.3f}m", 
                           (10, output_image.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.imshow("Distance Measurement", output_image)
        else:
            cv2.imshow("Distance Measurement", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and tags:
            # 记录测量结果
            actual_distance = input("请输入实际测量的距离(米): ")
            try:
                actual_distance = float(actual_distance)
                measured_distance = np.linalg.norm(tvec)
                error = abs(actual_distance - measured_distance)
                error_percent = error / actual_distance * 100
                
                result = {
                    'actual': actual_distance,
                    'measured': measured_distance,
                    'error': error,
                    'error_percent': error_percent
                }
                
                measurements.append(result)
                
                print(f"实际距离: {actual_distance:.3f}m")
                print(f"测量距离: {measured_distance:.3f}m")
                print(f"误差: {error:.3f}m ({error_percent:.2f}%)")
                print("结果已记录\n")
            except ValueError:
                print("输入格式错误，请输入数字\n")
                
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 显示测量结果总结
    if measurements:
        print("\n=== 距离测量测试结果 ===")
        print("实际距离(m)\t测量距离(m)\t误差(m)\t误差(%)")
        print("-" * 60)
        
        total_error_percent = 0
        for m in measurements:
            print(f"{m['actual']:.3f}\t\t{m['measured']:.3f}\t\t{m['error']:.3f}\t{m['error_percent']:.2f}")
            total_error_percent += m['error_percent']
        
        avg_error_percent = total_error_percent / len(measurements)
        print("-" * 60)
        print(f"平均误差百分比: {avg_error_percent:.2f}%")
        
        if avg_error_percent < 5:
            print("结论: 相机参数精度很好 (误差 < 5%)")
        elif avg_error_percent < 10:
            print("结论: 相机参数精度一般 (误差 5-10%)")
        else:
            print("结论: 相机参数精度较差 (误差 > 10%)，建议重新标定")
    
def visualize_coordinate_system(K, D):
    """可视化相机坐标系，用于理解相机的方向
    
    参数:
        K: 相机内参矩阵
        D: 畸变系数
    """
    # 创建一个虚拟的3D坐标系
    axis_length = 0.1  # 10cm
    virtual_pts = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
    
    # 相机位姿 (原点，无旋转)
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)
    
    # 打开相机
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开相机")
        return
    
    print("\n=== 坐标系可视化 ===")
    print("显示相机坐标系的方向，帮助理解相机的方向定义")
    print("红色 - X轴")
    print("绿色 - Y轴")
    print("蓝色 - Z轴")
    print("\n按'q'退出测试")
    
    while True:
        # 读取相机帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取相机帧")
            break
        
        # 创建全白图像作为背景
        background = np.ones_like(frame) * 255
        
        # 在图像中心创建一个坐标系
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 调整平移向量到图像中心
        tvec[0] = 0
        tvec[1] = 0
        tvec[2] = 0.5  # 0.5米距离，便于观察
        
        # 将3D点投影到图像平面
        imgpts, _ = cv2.projectPoints(virtual_pts, rvec, tvec, K, D)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        
        # 绘制坐标轴
        origin = tuple(imgpts[0])
        background = cv2.line(background, origin, tuple(imgpts[1]), (0, 0, 255), 5)  # X轴 - 红色
        background = cv2.line(background, origin, tuple(imgpts[2]), (0, 255, 0), 5)  # Y轴 - 绿色
        background = cv2.line(background, origin, tuple(imgpts[3]), (255, 0, 0), 5)  # Z轴 - 蓝色
        
        # 添加标签
        cv2.putText(background, "X", tuple(imgpts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(background, "Y", tuple(imgpts[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(background, "Z", tuple(imgpts[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 添加说明
        cv2.putText(background, "Camera Coordinate System", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 显示原始图像和坐标系可视化
        combined = np.hstack((frame, background))
        cv2.imshow("Coordinate System Visualization", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='相机参数验证工具')
    parser.add_argument('--camera_info', default='config/camera/camera_info_1.yaml',
                        help='相机参数文件路径 (默认: config/camera/camera_info_1.yaml)')
    args = parser.parse_args()
    
    # 读取相机参数
    try:
        print(f"读取相机参数: {args.camera_info}")
        K, D = read_camera_info(args.camera_info)
        print(f"相机内参矩阵K:\n{K}")
        print(f"畸变系数D: {D}")
    except Exception as e:
        print(f"相机参数读取错误: {e}")
        print("使用默认相机参数...")
        # 创建默认相机参数
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        D = np.zeros((4, 1), dtype=np.float32)
    
    # 显示主菜单
    while True:
        print("\n=== 相机参数验证工具 ===")
        print("1. 图像校正测试")
        print("2. 距离测量测试")
        print("3. 坐标系可视化")
        print("0. 退出")
        
        choice = input("\n请选择测试类型: ")
        
        if choice == '1':
            test_undistortion(K, D)
        elif choice == '2':
            tag_size = input("请输入AprilTag标签的实际大小(米，默认0.1): ") or 0.1
            test_distance_measurement(K, D, float(tag_size))
        elif choice == '3':
            visualize_coordinate_system(K, D)
        elif choice == '0':
            print("程序结束")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main() 