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

def undistort_image(image, camera_matrix, dist_coeffs, keep_fov=True, alpha=0.85):
    """图像畸变校正函数
    
    Args:
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        keep_fov: 是否保持视场角
        alpha: 视场保留比例
        
    Returns:
        校正后的图像
    """
    if image is None:
        print("警告: 输入图像为空，跳过畸变校正")
        return image
        
    h, w = image.shape[:2]
    
    try:
        if keep_fov:
            # 使用getOptimalNewCameraMatrix优化相机矩阵以保持视场角
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), alpha)
            
            # 校正图像
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
            
            # 如果alpha小于1.0，对图像进行裁剪以去除黑边
            if alpha < 1.0:
                x, y, w, h = roi
                if w > 0 and h > 0:  # 确保ROI有效
                    # 保证不超出边界
                    x = max(0, x)
                    y = max(0, y)
                    if x + w > undistorted.shape[1]:
                        w = undistorted.shape[1] - x
                    if y + h > undistorted.shape[0]:
                        h = undistorted.shape[0] - y
                        
                    if w > 0 and h > 0:  # 再次检查裁剪后的尺寸
                        undistorted = undistorted[y:y+h, x:x+w]
                        undistorted = cv2.resize(undistorted, (image.shape[1], image.shape[0]))
        else:
            # 使用原始相机矩阵进行校正
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
            
        return undistorted
    except Exception as e:
        print(f"畸变校正错误: {e}")
        return image  # 发生错误时返回原始图像

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
    print(f"畸变校正: {'已开启' if camera_config.undistort else '已关闭'}")
    if camera_config.undistort:
        print(f"视场保持: {'已开启' if camera_config.keep_fov else '已关闭'}, alpha={camera_config.alpha}")
    
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
    print("按 'q' 键退出，按 'i' 键手动初始化系统，按 'k' 键切换卡尔曼滤波，按 'p' 键切换平面约束")
    if camera_config.undistort:
        print("按 'u' 键切换畸变校正，按 'f' 键切换视场保持，按 '+/-' 键调整alpha值")
    
    # 畸变校正控制变量
    undistort_enabled = camera_config.undistort
    keep_fov = camera_config.keep_fov
    alpha = camera_config.alpha
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取相机帧")
                break
            
            # 应用畸变校正（如果启用）
            if undistort_enabled:
                try:
                    corrected_frame = undistort_image(frame, K, D, keep_fov, alpha)
                    if corrected_frame is not None and corrected_frame.shape == frame.shape:
                        frame = corrected_frame
                    else:
                        print("警告: 畸变校正返回的图像无效，使用原始图像")
                except Exception as e:
                    print(f"畸变校正异常: {e}")
                    # 发生异常时继续使用原始图像
            
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
            
            # 添加畸变校正信息到图像
            if undistort_enabled:
                cv2.putText(output_image, 
                           f"畸变校正: 开启 | 视场保持: {'开启' if keep_fov else '关闭'} | alpha: {alpha:.2f}", 
                           (10, output_image.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
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
                        # 应用畸变校正（如果启用）
                        if undistort_enabled:
                            try:
                                corrected_frame = undistort_image(init_frame, K, D, keep_fov, alpha)
                                if corrected_frame is not None and corrected_frame.shape == init_frame.shape:
                                    init_frame = corrected_frame
                                else:
                                    print("警告: 初始化时畸变校正返回无效图像，使用原始图像")
                            except Exception as e:
                                print(f"初始化时畸变校正异常: {e}")
                                # 发生异常时继续使用原始图像
                        success = tracker.initialize_system(init_frame)
                        if success:
                            print("手动初始化成功！")
                        else:
                            print("手动初始化失败，请确保所有标签可见。")
                elif key == ord('k'):
                    # 切换卡尔曼滤波状态
                    tracker.use_kalman = not tracker.use_kalman
                    print(f"卡尔曼滤波已{'开启' if tracker.use_kalman else '关闭'}")
                elif key == ord('p'):
                    # 切换平面约束状态
                    tracker.apply_plane_constraint = not tracker.apply_plane_constraint
                    print(f"平面约束已{'开启' if tracker.apply_plane_constraint else '关闭'}")
                elif key == ord('u'):
                    # 切换畸变校正状态
                    undistort_enabled = not undistort_enabled
                    print(f"畸变校正已{'开启' if undistort_enabled else '关闭'}")
                elif key == ord('f'):
                    # 切换视场保持状态
                    keep_fov = not keep_fov
                    print(f"视场保持已{'开启' if keep_fov else '关闭'}")
                elif key == ord('+') or key == ord('='):
                    # 增加alpha值
                    alpha = min(1.0, alpha + 0.05)
                    print(f"视场比例 alpha: {alpha:.2f}")
                elif key == ord('-') or key == ord('_'):
                    # 减小alpha值
                    alpha = max(0.0, alpha - 0.05)
                    print(f"视场比例 alpha: {alpha:.2f}")
    
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