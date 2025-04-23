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
    --keep_fov: 保持视场角，防止校正后图像缩小
    --alpha: 视场保留比例 (0.0-2.0), 用于控制去除黑边程度 (默认: 1.0, 步长: 0.05)
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

def visualize_single_image(image_path, camera_matrix, dist_coeffs, image_size, show_grid=False, enhance=False, keep_fov=False, alpha=1.0):
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
            
        # 根据是否保持视场角选择相应的相机矩阵
        if keep_fov:
            # 优化相机矩阵以保持视场角
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, enhanced_dist, (image_size[0], image_size[1]), alpha)
            
            # 校正图像
            undistorted = cv2.undistort(img, camera_matrix, enhanced_dist, None, new_camera_matrix)
            
            # 裁剪图像（可选）
            if alpha < 1.0:
                x, y, w, h = roi
                undistorted = undistorted[y:y+h, x:x+w]
                undistorted = cv2.resize(undistorted, (image_size[0], image_size[1]))
        else:
            # 使用原始相机矩阵进行校正
            undistorted = cv2.undistort(img, camera_matrix, enhanced_dist)
        
        # 生成差异图像
        diff = cv2.absdiff(img, undistorted)
        diff = cv2.convertScaleAbs(diff, alpha=5.0)  # 放大差异
        
        # 创建带标签的图像
        cv2.putText(img, "原始图像", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(undistorted, f"校正图像{' (保持视场)' if keep_fov else ''}", (10, 30), 
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

def visualize_camera(camera_id, camera_matrix, dist_coeffs, image_size, show_grid=False, enhance=False, keep_fov=False, alpha=1.0):
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
        
        # 计算适应所有alpha值的最大黑边ROI（alpha=0时）
        temp_matrix, full_roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, enhanced_dist, (image_size[0], image_size[1]), 0)
        
        # 保存原始ROI信息
        full_roi_x, full_roi_y, full_roi_w, full_roi_h = full_roi
        
        # 生成重映射表（根据是否保持视场角）
        if keep_fov:
            # 使用getOptimalNewCameraMatrix优化相机矩阵
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, enhanced_dist, (image_size[0], image_size[1]), alpha)
            
            # 生成重映射表
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix, enhanced_dist, None, new_camera_matrix,
                (image_size[0], image_size[1]), cv2.CV_32FC1)
            
            # 保存ROI以便裁剪（始终使用alpha=0时的完整ROI）
            roi = full_roi
            has_roi = True
        else:
            # 使用原始相机矩阵
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix, enhanced_dist, None, camera_matrix,
                (image_size[0], image_size[1]), cv2.CV_32FC1)
            has_roi = False
            
        # alpha值调整步长
        alpha_step = 0.05  # 更小的步长
        # alpha值范围扩大
        alpha_min = 0.0  # 允许从0开始
        alpha_max = 2.0  # 允许超过1以提供更广的调整范围
        
        # 实时显示的缩放比例
        zoom_factor = 1.0
        
        print("按 'q' 退出, 'g' 切换网格, 'e' 切换增强效果, 'f' 切换视场保持")
        print("a/d: 减小/增加alpha值, z/x: 减小/增加缩放比例")
        show_grid_current = show_grid
        show_enhanced = enhance
        keep_fov_current = keep_fov
        alpha_current = alpha
        
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
            
            # 如果启用ROI裁剪，实现平滑渐变的裁剪过程
            if keep_fov_current and has_roi:
                # 获取完整ROI信息
                x, y, w, h = full_roi
                
                if alpha_current < 1.0 and w > 0 and h > 0:
                    # 根据alpha计算裁剪比例（alpha=1时不裁剪，alpha=0时完全裁剪）
                    crop_ratio = 1.0 - alpha_current
                    
                    # 计算新的裁剪区域（从全尺寸向ROI过渡）
                    # 图像宽度从image_size[0]向w过渡
                    # 图像高度从image_size[1]向h过渡
                    new_w = int(image_size[0] - (image_size[0] - w) * crop_ratio)
                    new_h = int(image_size[1] - (image_size[1] - h) * crop_ratio)
                    
                    # 计算新的起始位置（从0,0向x,y过渡）
                    new_x = int(x * crop_ratio)
                    new_y = int(y * crop_ratio)
                    
                    # 裁剪并缩放
                    if new_w > 0 and new_h > 0 and new_x + new_w <= image_size[0] and new_y + new_h <= image_size[1]:
                        undistorted = undistorted[new_y:new_y+new_h, new_x:new_x+new_w]
                        undistorted = cv2.resize(undistorted, (image_size[0], image_size[1]))
            
            # 应用缩放 (用于在不改变相机矩阵的情况下调整视场大小)
            if zoom_factor != 1.0:
                h, w = undistorted.shape[:2]
                # 计算缩放尺寸
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                # 计算裁剪的起始点
                start_y = max(0, (h - new_h) // 2)
                start_x = max(0, (w - new_w) // 2)
                
                if zoom_factor < 1.0:  # 缩小 (显示更多区域)
                    # 创建更大的画布
                    canvas = np.zeros((h, w, 3), dtype=undistorted.dtype)
                    # 计算小图像的位置
                    pos_y = (h - new_h) // 2
                    pos_x = (w - new_w) // 2
                    # 缩放图像
                    small_img = cv2.resize(undistorted, (new_w, new_h))
                    # 放到画布中心
                    canvas[pos_y:pos_y+new_h, pos_x:pos_x+new_w] = small_img
                    undistorted = canvas
                else:  # 放大 (只显示中心区域)
                    # 确保不超出图像边界
                    if new_h < h and new_w < w:
                        # 裁剪中心区域
                        center_y, center_x = h // 2, w // 2
                        half_new_h, half_new_w = new_h // 2, new_w // 2
                        # 裁剪区域
                        crop_y1 = max(0, center_y - half_new_h)
                        crop_y2 = min(h, center_y + half_new_h)
                        crop_x1 = max(0, center_x - half_new_w)
                        crop_x2 = min(w, center_x + half_new_w)
                        # 裁剪并缩放回原始大小
                        cropped = undistorted[crop_y1:crop_y2, crop_x1:crop_x2]
                        undistorted = cv2.resize(cropped, (w, h))
            
            # 生成差异图像
            diff = cv2.absdiff(frame, undistorted)
            diff = cv2.convertScaleAbs(diff, alpha=5.0)
            
            # 添加标签
            cv2.putText(frame, "原始图像", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(undistorted, 
                       f"校正图像 {'(增强)' if show_enhanced else ''} {'(保持视场)' if keep_fov_current else ''}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(diff, "差异图像 (x5)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                       
            # 创建信息面板
            info_panel = np.zeros((100, image_size[0]*2, 3), dtype=np.uint8)
            cv2.putText(info_panel, "q: 退出  g: 网格  e: 增强  f: 保持视场  a/d: alpha值  z/x: 缩放", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(info_panel, 
                       f"网格: {'开启' if show_grid_current else '关闭'}  " +
                       f"增强: {'开启' if show_enhanced else '关闭'}  " +
                       f"保持视场: {'开启' if keep_fov_current else '关闭'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(info_panel, 
                       f"alpha: {alpha_current:.2f}  缩放: {zoom_factor:.2f}x",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
                update_mapping = True
            elif key == ord('f'):
                keep_fov_current = not keep_fov_current
                print(f"保持视场: {'开启' if keep_fov_current else '关闭'}")
                update_mapping = True
            elif key == ord('a'):
                # 减小alpha值 (更多裁剪)
                alpha_current = max(alpha_min, alpha_current - alpha_step)
                print(f"视场比例 alpha: {alpha_current:.2f}")
                update_mapping = True
            elif key == ord('d'):
                # 增加alpha值 (更少裁剪)
                alpha_current = min(alpha_max, alpha_current + alpha_step)
                print(f"视场比例 alpha: {alpha_current:.2f}")
                update_mapping = True
            elif key == ord('z'):
                # 减小缩放比例 (查看更广视场)
                zoom_factor = max(0.5, zoom_factor - 0.05)
                print(f"缩放比例: {zoom_factor:.2f}x")
            elif key == ord('x'):
                # 增加缩放比例 (放大查看细节)
                zoom_factor = min(2.0, zoom_factor + 0.05)
                print(f"缩放比例: {zoom_factor:.2f}x")
            else:
                continue  # 如果没有按键需要处理，跳过重新计算映射
                
            # 处理需要更新映射的按键
            if key in [ord('e'), ord('f'), ord('a'), ord('d')]:
                # 准备畸变系数
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
                
                # 根据是否保持视场角更新映射
                if keep_fov_current:
                    # 使用getOptimalNewCameraMatrix优化相机矩阵
                    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                        camera_matrix, enhanced_dist, (image_size[0], image_size[1]), alpha_current)
                    
                    # 更新重映射表
                    map1, map2 = cv2.initUndistortRectifyMap(
                        camera_matrix, enhanced_dist, None, new_camera_matrix,
                        (image_size[0], image_size[1]), cv2.CV_32FC1)
                    
                    # 使用完整ROI以便平滑过渡
                    roi = full_roi
                    has_roi = True
                else:
                    # 使用原始相机矩阵更新映射
                    map1, map2 = cv2.initUndistortRectifyMap(
                        camera_matrix, enhanced_dist, None, camera_matrix,
                        (image_size[0], image_size[1]), cv2.CV_32FC1)
                    has_roi = False
        
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
    parser.add_argument('--keep_fov', action='store_true',
                        help='保持视场角，防止校正后图像缩小')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='视场保留比例 (0.0-2.0), 用于控制去除黑边程度 (默认: 1.0, 步长: 0.05)')
    args = parser.parse_args()
    
    # 更新alpha值限制范围
    if args.alpha < 0.0:
        print(f"警告: alpha值 {args.alpha} 太小，已调整为最小值 0.0")
        args.alpha = 0.0
    elif args.alpha > 2.0:
        print(f"警告: alpha值 {args.alpha} 太大，已调整为最大值 2.0")
        args.alpha = 2.0
    
    # 加载相机标定文件
    camera_matrix, dist_coeffs, image_size = load_camera_calibration(args.calibration)
    
    # 根据输入选择可视化模式
    if args.image:
        visualize_single_image(args.image, camera_matrix, dist_coeffs, image_size, 
                              args.grid, args.enhance, args.keep_fov, args.alpha)
    else:
        visualize_camera(args.camera, camera_matrix, dist_coeffs, image_size, 
                        args.grid, args.enhance, args.keep_fov, args.alpha)

if __name__ == "__main__":
    main() 