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
from calibration import save_camera_calibration
from filesystem import create_dirs_if_not_exist

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
                        print(f"样本位置差异太小 ({mean_dist:.2f} 像素)，忽略")
                    else:
                        print(f"样本差异度: {mean_dist:.2f} 像素")
                
                    # 简单的姿态估计，检查样本角度分布
                    if len(objpoints) >= 3:
                        # 使用solvePnP估计相机姿态
                        # 创建一个临时相机矩阵用于姿态估计
                        temp_camera_matrix = np.array([
                            [actual_width, 0, actual_width/2],
                            [0, actual_width, actual_height/2],
                            [0, 0, 1]
                        ], dtype=np.float32)
                        
                        _, rvec, tvec = cv2.solvePnP(
                            objp, corners2, cameraMatrix=temp_camera_matrix, distCoeffs=None, 
                            flags=cv2.SOLVEPNP_ITERATIVE)
                        
                        # 将旋转向量转换为欧拉角
                        R, _ = cv2.Rodrigues(rvec)
                        euler_angles = np.degrees(cv2.RQDecomp3x3(R)[0])
                        
                        # 检查与已有样本的角度差异
                        angle_too_similar = False
                        for i in range(len(objpoints)):
                            # 为已有样本计算姿态
                            _, r, _ = cv2.solvePnP(
                                objp, imgpoints[i], cameraMatrix=temp_camera_matrix, distCoeffs=None, 
                                flags=cv2.SOLVEPNP_ITERATIVE)
                            R_prev, _ = cv2.Rodrigues(r)
                            euler_prev = np.degrees(cv2.RQDecomp3x3(R_prev)[0])
                            
                            # 计算角度差异
                            angle_diff = np.linalg.norm(euler_angles - euler_prev)
                            if angle_diff < 15.0:  # 角度阈值（度）
                                angle_too_similar = True
                                print(f"与样本 {i+1} 角度太相似 ({angle_diff:.2f}°)，建议从不同角度拍摄")
                                break
                        
                        # 如果角度差异太小，给出警告但仍然保存
                        if angle_too_similar:
                            print("警告: 建议从更多不同角度拍摄棋盘格以提高标定质量")
                
                if save_sample:
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    num_samples += 1
                    last_detection_time = current_time
                    print(f"自动采集样本 {num_samples}/{args.samples}")
                    
                    # 增加视觉反馈
                    # cv2.putText(display_frame, "Sample collected!", (actual_width//2-100, actual_height//2),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow('相机标定', display_frame)
                    cv2.waitKey(500)
                    
        
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
    
    # 计算每个棋盘格的重投影误差
    mean_errors = []
    max_errors = []
    error_imgs = []
    
    print("\n计算各样本的重投影误差...")
    for i in range(len(objpoints)):
        # 计算投影点
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
        
        # 计算误差
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_errors.append(error)
        
        # 计算每个点的误差
        point_errors = []
        for j in range(len(imgpoints[i])):
            pt1 = tuple(imgpoints[i][j][0])
            pt2 = tuple(imgpoints2[j][0])
            err = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            point_errors.append(err)
        
        max_errors.append(max(point_errors))
        
        print(f"样本 {i+1}: 平均误差 = {error:.6f} 像素, 最大误差 = {max(point_errors):.6f} 像素")
        
        # 创建误差可视化图像(可选，仅在debug模式下)
        if len(mean_errors) <= 5:  # 只保存前5个图像以节省空间
            # 获取原始图像尺寸，创建空白图像
            img = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)
            
            # 绘制原始角点(红色)和重投影角点(绿色)
            for j in range(len(imgpoints[i])):
                pt1 = tuple(map(int, imgpoints[i][j][0]))
                pt2 = tuple(map(int, imgpoints2[j][0]))
                
                # 原始检测点(红色)
                cv2.circle(img, pt1, 5, (0, 0, 255), -1)
                
                # 重投影点(绿色)
                cv2.circle(img, pt2, 3, (0, 255, 0), -1)
                
                # 连线(蓝色)
                cv2.line(img, pt1, pt2, (255, 0, 0), 1)
                
                # 标记点编号
                cv2.putText(img, str(j), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 标记误差值
                err_text = f"{point_errors[j]:.2f}"
                cv2.putText(img, err_text, (pt2[0]+5, pt2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 添加样本信息
            cv2.putText(img, f"样本 {i+1}, 平均误差: {error:.2f}px", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            error_imgs.append(img)
    
    # 计算重投影误差统计
    mean_error = np.mean(mean_errors)
    max_mean_error = np.max(mean_errors)
    max_error = np.max(max_errors)
    std_error = np.std(mean_errors)
    
    print(f"\n重投影误差统计:")
    print(f"  平均误差: {mean_error:.6f} 像素")
    print(f"  最大平均误差: {max_mean_error:.6f} 像素 (样本 {np.argmax(mean_errors)+1})")
    print(f"  最大误差: {max_error:.6f} 像素")
    print(f"  误差标准差: {std_error:.6f} 像素")
    
    # 保存重投影误差图像
    if error_imgs:
        print("\n保存重投影误差可视化...")
        output_dir = os.path.join(os.path.dirname(args.output), "error_visualization")
        create_dirs_if_not_exist(output_dir)
        
        for i, img in enumerate(error_imgs):
            output_path = os.path.join(output_dir, f"reproj_error_sample_{i+1}.png")
            cv2.imwrite(output_path, img)
        
        print(f"已保存重投影误差可视化图像到 {output_dir}")
        
    # 保存重投影误差信息
    error_data = {
        'mean_reprojection_error': float(ret),
        'sample_errors': {
            'mean_errors': [float(e) for e in mean_errors],
            'max_errors': [float(e) for e in max_errors]
        },
        'error_statistics': {
            'mean': float(mean_error),
            'max_mean': float(max_mean_error),
            'max': float(max_error),
            'std': float(std_error)
        }
    }
    
    # 保存标定结果
    calibration_data = save_camera_calibration(args.output, camera_matrix, dist_coefs, actual_width, actual_height, error_data)
    
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
        
        # 放大畸变系数以增强效果（仅用于可视化）
        enhanced_dist_coefs = dist_coefs.copy()
        # 放大径向畸变系数（k1, k2, k3）
        scaling_factor = 1.5  # 可调整此值以增强效果
        # 仅当畸变不为零时应用缩放
        if np.abs(enhanced_dist_coefs[0, 0]) > 1e-5:
            enhanced_dist_coefs[0, 0] *= scaling_factor  # k1
        if np.abs(enhanced_dist_coefs[0, 1]) > 1e-5:
            enhanced_dist_coefs[0, 1] *= scaling_factor  # k2
        if len(enhanced_dist_coefs[0]) > 4 and np.abs(enhanced_dist_coefs[0, 4]) > 1e-5:
            enhanced_dist_coefs[0, 4] *= scaling_factor  # k3
        
        # 增强效果的校正映射
        enhanced_map1, enhanced_map2 = cv2.initUndistortRectifyMap(
            camera_matrix, enhanced_dist_coefs, None, camera_matrix, 
            (actual_width, actual_height), cv2.CV_32FC1)
            
        # 绘制参考线以便更容易观察校正效果
        def draw_grid(img, grid_size=50):
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
        
        print("按'q'退出预览，按'e'切换增强效果，按'g'切换网格显示")
        
        # 展示模式标志
        show_enhanced = False
        show_grid = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 添加网格线（如果启用）
            if show_grid:
                frame = draw_grid(frame)
            
            # 未校正的图像
            cv2.putText(frame, "original", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 根据选择使用正常校正或增强校正
            if show_enhanced:
                undistorted = cv2.remap(frame, enhanced_map1, enhanced_map2, cv2.INTER_LINEAR)
                cv2.putText(undistorted, "correct (enhance)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                cv2.putText(undistorted, "correct ", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 如果启用网格，也在校正图像上添加
            if show_grid:
                undistorted = draw_grid(undistorted)
            
            # 计算差异图像以突出校正的变化
            diff = cv2.absdiff(frame, undistorted)
            # 增强差异使其更明显
            diff = cv2.convertScaleAbs(diff, alpha=5.0)
            cv2.putText(diff, "差异图像 (x5)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 创建一个信息面板
            info_panel = np.zeros((100, actual_width*2, 3), dtype=np.uint8)
            cv2.putText(info_panel, "q: quit  e: enhance  g: grid", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_panel, f"增强效果: {'开启' if show_enhanced else '关闭'}  网格: {'开启' if show_grid else '关闭'}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 并排显示
            top_row = np.hstack((frame, undistorted))
            bottom_row = np.hstack((diff, np.zeros_like(diff)))  # 右下方留空
            
            # 组合所有图像
            combined = np.vstack((top_row, bottom_row, info_panel))
            cv2.imshow('校正效果对比', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                show_enhanced = not show_enhanced
                print(f"增强效果: {'开启' if show_enhanced else '关闭'}")
            elif key == ord('g'):
                show_grid = not show_grid
                print(f"网格显示: {'开启' if show_grid else '关闭'}")
                
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n标定过程完成!")
    print(f"标定结果已保存至: {args.output}")
    print("\n您可以使用以下工具检查和可视化标定结果:")
    print(f"1. 检查标定质量: python lib/check_calibration.py --calibration {args.output}")
    print(f"2. 可视化标定效果: python lib/visualize_calibration.py --calibration {args.output} [--enhance] [--grid]")
    print("   - 添加 --enhance 参数可增强视觉效果")
    print("   - 添加 --grid 参数可显示网格线")
    print("   - 添加 --image <图像路径> 可对静态图像进行校正")
    print("\n如果校正效果不明显，可能的原因包括:")
    print("- 相机畸变本身很小")
    print("- 标定样本不够多或分布不均匀")
    print("- 标定时的方格大小参数和实际不符")
    print("- 棋盘格未能覆盖图像边缘区域(畸变通常在边缘更显著)")
    
if __name__ == "__main__":
    main() 