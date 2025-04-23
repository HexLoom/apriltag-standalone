#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
相机标定可视化工具

读取相机标定文件并可视化畸变校正效果。
可以加载图像文件或使用相机实时预览效果。

使用方法:
    python visualize_calibration.py [--calibration CALIBRATION_FILE] [--camera CAMERA_ID] [--image IMAGE_FILE]

参数:
    --calibration: 相机标定文件路径 (默认: config/camera/HSK_200W53_1080P.yaml)
    --camera: 相机设备ID (默认: 0)
    --image: 可选，用于测试校正的图像文件
    --grid: 显示网格线以更好地观察校正效果
    --enhance: 增强畸变系数以更明显地观察效果
"""

import os
import sys
import argparse
import numpy as np
import cv2
import yaml
import time

# 检查必要的依赖项
def check_dependencies():
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
        
    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")
    
    if missing_deps:
        print("警告: 缺少以下依赖库:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\n请使用以下命令安装:")
        print(f"  pip install {' '.join(missing_deps)}")
        
        if missing_deps:  # 如果缺少任何核心依赖
            print("错误: 缺少核心依赖项，程序无法继续运行。")
            sys.exit(1)
    
    return True

def load_camera_calibration(filename):
    """加载相机标定文件"""
    print(f"加载标定文件: {filename}")
    try:
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
            
        # 提取相机矩阵
        camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        
        # 提取畸变系数
        dist_coeffs = np.array(data['distortion_coefficients']['data']).reshape(
            data['distortion_coefficients']['rows'],
            data['distortion_coefficients']['cols']
        )
        
        # 获取图像尺寸
        image_width = data['image_width']
        image_height = data['image_height']
        
        print(f"图像尺寸: {image_width}x{image_height}")
        print(f"相机矩阵:\n{camera_matrix}")
        print(f"畸变系数: {dist_coeffs.ravel()}")
        
        return camera_matrix, dist_coeffs, (image_width, image_height)
    except Exception as e:
        print(f"加载标定文件错误: {e}")
        sys.exit(1)

def draw_grid(img, grid_size=50):
    """在图像上绘制网格线"""
    h, w = img.shape[:2]
    grid_img = img.copy()
    
    # 绘制水平线
    for y in range(0, h, grid_size):
        cv2.line(grid_img, (0, y), (w, y), (0, 255, 255), 1)
        
    # 绘制垂直线
    for x in range(0, w, grid_size):
        cv2.line(grid_img, (x, 0), (x, h), (0, 255, 255), 1)
        
    # 绘制中心十字线
    cv2.line(grid_img, (w//2, 0), (w//2, h), (0, 0, 255), 2)
    cv2.line(grid_img, (0, h//2), (w, h//2), (0, 0, 255), 2)
    
    return grid_img

def visualize_single_image(image_path, camera_matrix, dist_coeffs, image_size, show_grid=False, enhance=False):
    """可视化单张图像的校正效果"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
            
        # 缩放图像到标定尺寸
        img = cv2.resize(img, (image_size[0], image_size[1]))
        
        # 准备畸变系数
        if enhance:
            enhanced_dist = dist_coeffs.copy()
            scaling_factor = 1.5
            if np.abs(enhanced_dist[0, 0]) > 1e-5:
                enhanced_dist[0, 0] *= scaling_factor  # k1
            if np.abs(enhanced_dist[0, 1]) > 1e-5:
                enhanced_dist[0, 1] *= scaling_factor  # k2
            if enhanced_dist.shape[1] > 4 and np.abs(enhanced_dist[0, 4]) > 1e-5:
                enhanced_dist[0, 4] *= scaling_factor  # k3
        else:
            enhanced_dist = dist_coeffs
            
        # 添加网格
        if show_grid:
            img = draw_grid(img)
            
        # 校正图像
        undistorted = cv2.undistort(img, camera_matrix, enhanced_dist)
        
        # 生成差异图像
        diff = cv2.absdiff(img, undistorted)
        diff = cv2.convertScaleAbs(diff, alpha=5.0)  # 放大差异
        
        # 创建带标签的图像
        cv2.putText(img, "原始图像", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(undistorted, "校正图像", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(diff, "差异图像 (x5)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                   
        # 组合显示
        top_row = np.hstack((img, undistorted))
        bottom_row = np.hstack((diff, np.zeros_like(diff)))
        combined = np.vstack((top_row, bottom_row))
        
        # 显示
        cv2.imshow("相机校正效果", combined)
        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"处理图像错误: {e}")

def visualize_camera(camera_id, camera_matrix, dist_coeffs, image_size, show_grid=False, enhance=False):
    """实时可视化相机校正效果"""
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"无法打开相机 {camera_id}")
            return
            
        # 设置相机分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])
        
        # 检查实际分辨率
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != image_size[0] or actual_height != image_size[1]:
            print(f"警告: 相机分辨率与标定不符，预览可能不准确")
            print(f"标定: {image_size[0]}x{image_size[1]}, 实际: {actual_width}x{actual_height}")
        
        # 准备畸变系数（可能增强）
        if enhance:
            enhanced_dist = dist_coeffs.copy()
            scaling_factor = 1.5
            if np.abs(enhanced_dist[0, 0]) > 1e-5:
                enhanced_dist[0, 0] *= scaling_factor
            if np.abs(enhanced_dist[0, 1]) > 1e-5:
                enhanced_dist[0, 1] *= scaling_factor
            if enhanced_dist.shape[1] > 4 and np.abs(enhanced_dist[0, 4]) > 1e-5:
                enhanced_dist[0, 4] *= scaling_factor
        else:
            enhanced_dist = dist_coeffs
            
        # 生成重映射表
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, enhanced_dist, None, camera_matrix,
            (image_size[0], image_size[1]), cv2.CV_32FC1)
            
        print("按 'q' 退出, 'g' 切换网格, 'e' 切换增强效果")
        show_grid_current = show_grid
        show_enhanced = enhance
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 调整帧大小
            frame = cv2.resize(frame, (image_size[0], image_size[1]))
                
            # 添加网格线
            if show_grid_current:
                frame = draw_grid(frame)
                
            # 校正图像
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            
            # 生成差异图像
            diff = cv2.absdiff(frame, undistorted)
            diff = cv2.convertScaleAbs(diff, alpha=5.0)
            
            # 添加标签
            cv2.putText(frame, "原始图像", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(undistorted, 
                       f"校正图像 {'(增强)' if show_enhanced else ''}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(diff, "差异图像 (x5)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                       
            # 创建信息面板
            info_panel = np.zeros((100, image_size[0]*2, 3), dtype=np.uint8)
            cv2.putText(info_panel, "q: 退出  g: 网格  e: 增强效果", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_panel, 
                       f"网格: {'开启' if show_grid_current else '关闭'}  增强: {'开启' if show_enhanced else '关闭'}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 组合显示
            top_row = np.hstack((frame, undistorted))
            bottom_row = np.hstack((diff, np.zeros_like(diff)))
            combined = np.vstack((top_row, bottom_row, info_panel))
            
            cv2.imshow("相机校正效果", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                show_grid_current = not show_grid_current
                print(f"网格显示: {'开启' if show_grid_current else '关闭'}")
            elif key == ord('e'):
                show_enhanced = not show_enhanced
                print(f"增强效果: {'开启' if show_enhanced else '关闭'}")
                # 重新计算映射
                if show_enhanced:
                    enhanced_dist = dist_coeffs.copy()
                    scaling_factor = 1.5
                    if np.abs(enhanced_dist[0, 0]) > 1e-5:
                        enhanced_dist[0, 0] *= scaling_factor
                    if np.abs(enhanced_dist[0, 1]) > 1e-5:
                        enhanced_dist[0, 1] *= scaling_factor
                    if enhanced_dist.shape[1] > 4 and np.abs(enhanced_dist[0, 4]) > 1e-5:
                        enhanced_dist[0, 4] *= scaling_factor
                else:
                    enhanced_dist = dist_coeffs
                
                # 更新映射
                map1, map2 = cv2.initUndistortRectifyMap(
                    camera_matrix, enhanced_dist, None, camera_matrix,
                    (image_size[0], image_size[1]), cv2.CV_32FC1)
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"相机预览错误: {e}")

def main():
    # 检查依赖项
    check_dependencies()
    
    parser = argparse.ArgumentParser(description='相机标定可视化工具')
    parser.add_argument('--calibration', type=str, default='config/camera/HSK_200W53_1080P.yaml',
                        help='相机标定文件路径 (默认: config/camera/HSK_200W53_1080P.yaml)')
    parser.add_argument('--camera', type=int, default=0,
                        help='相机设备ID (默认: 0)')
    parser.add_argument('--image', type=str, default=None,
                        help='可选，用于测试校正的图像文件')
    parser.add_argument('--grid', action='store_true',
                        help='显示网格线以更好地观察校正效果')
    parser.add_argument('--enhance', action='store_true',
                        help='增强畸变系数以更明显地观察效果')
    args = parser.parse_args()
    
    # 加载相机标定文件
    camera_matrix, dist_coeffs, image_size = load_camera_calibration(args.calibration)
    
    # 根据输入选择可视化模式
    if args.image:
        visualize_single_image(args.image, camera_matrix, dist_coeffs, image_size, args.grid, args.enhance)
    else:
        visualize_camera(args.camera, camera_matrix, dist_coeffs, image_size, args.grid, args.enhance)

if __name__ == "__main__":
    main() 