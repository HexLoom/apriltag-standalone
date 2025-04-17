#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Desktop AprilTag Tracking System

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
import json
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入AprilTag检测器
from lib.apriltag import Detector, DetectorOptions, draw_detection_results

# ================== 工具函数 ==================
def read_json(file_path):
    """读取JSON配置文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def read_camera_info(yaml_path):
    """从OpenCV YAML文件读取相机参数"""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()
    return K, D

def create_dirs_if_not_exist(path):
    """创建目录（如果不存在）"""
    if not os.path.exists(path):
        os.makedirs(path)

def matrix_to_quaternion(R):
    """将旋转矩阵转换为四元数"""
    m00 = R[0,0]; m01 = R[0,1]; m02 = R[0,2]
    m10 = R[1,0]; m11 = R[1,1]; m12 = R[1,2]
    m20 = R[2,0]; m21 = R[2,1]; m22 = R[2,2]

    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S 
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S 
        qz = (m02 + m20) / S
    elif (m11 > m22):
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S 
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw])

# ================== 配置类 ==================
class AprilTagConfig:
    def __init__(self, family, size, threads, max_hamming, z_up, decimate, blur, refine_edges, debug):
        self.family = family
        self.size = size          # 标签物理尺寸（米）
        self.threads = threads
        self.max_hamming = max_hamming
        self.z_up = z_up          # 是否Z轴向上
        self.decimate = decimate
        self.blur = blur
        self.refine_edges = refine_edges
        self.debug = debug

class ArchiveConfig:
    def __init__(self, enable, preview, save_raw, save_pred, preview_delay, path):
        self.enable = enable
        self.preview = preview
        self.save_raw = save_raw
        self.save_pred = save_pred
        self.preview_delay = preview_delay
        self.path = path

class TableConfig:
    """桌面配置"""
    def __init__(self, reference_tags, moving_tag, tag_positions):
        self.reference_tags = reference_tags  # 参考标签ID列表
        self.moving_tag = moving_tag          # 移动标签ID
        self.tag_positions = {}               # 参考标签预设位置
        # 将字符串键转为整数
        for tag_id, position in tag_positions.items():
            self.tag_positions[int(tag_id)] = position

# ================== 可视化类 ==================
class TableVisualizer:
    def __init__(self):
        # 初始化坐标范围和图形存储变量
        # 自动调整坐标范围
        self.x_min = -0.2
        self.x_max = 0.7
        self.y_min = -0.2
        self.y_max = 0.7
        self.z_min = -0.2
        self.z_max = 0.5
        
        # 是否需要更新坐标范围
        self.needs_axis_update = False
        
        # 存储图形对象
        self.quivers = []
        self.texts = []
        self.scatter_points = []
        self.table_lines = []
        
        # 标记移动轨迹
        self.track_history = []
        self.track_line = None
        
        # 创建绘图窗口和3D坐标系
        plt.ion()  # 启用交互模式
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_axes()

    def setup_axes(self):
        """初始化3D坐标系"""
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max) 
        self.ax.set_zlim(self.z_min, self.z_max)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('AprilTag Table Tracking')
        self.ax.view_init(elev=30, azim=45)
        
        # 绘制世界坐标系
        self.ax.quiver(0, 0, 0, 0.1, 0, 0, color='r', linewidth=2)  # X轴
        self.ax.quiver(0, 0, 0, 0, 0.1, 0, color='g', linewidth=2)  # Y轴
        self.ax.quiver(0, 0, 0, 0, 0, 0.1, color='b', linewidth=2)  # Z轴

    def update_axis_limits(self, tag_poses):
        """根据检测到的标签自动调整坐标范围"""
        if not tag_poses:
            return
            
        # 收集所有标签位置
        positions = [t for t, _ in tag_poses.values()]
        if not positions:
            return
            
        # 计算当前边界
        pos_array = np.array(positions)
        x_min_current = np.min(pos_array[:, 0]) - 0.3
        x_max_current = np.max(pos_array[:, 0]) + 0.3
        y_min_current = np.min(pos_array[:, 1]) - 0.3
        y_max_current = np.max(pos_array[:, 1]) + 0.3
        z_min_current = np.min(pos_array[:, 2]) - 0.2
        z_max_current = np.max(pos_array[:, 2]) + 0.3
        
        # 检查是否需要更新
        if (x_min_current < self.x_min or x_max_current > self.x_max or
            y_min_current < self.y_min or y_max_current > self.y_max or
            z_min_current < self.z_min or z_max_current > self.z_max):
            
            # 更新范围（稍微扩大一点）
            self.x_min = min(self.x_min, x_min_current)
            self.x_max = max(self.x_max, x_max_current)
            self.y_min = min(self.y_min, y_min_current)
            self.y_max = max(self.y_max, y_max_current)
            self.z_min = min(self.z_min, z_min_current)
            self.z_max = max(self.z_max, z_max_current)
            
            # 设置需要更新标志
            self.needs_axis_update = True

    def draw_table(self, reference_tags_poses):
        """绘制桌面 - 根据实际检测到的参考标签绘制，不固定顺序"""
        # 清除旧的桌面线
        while self.table_lines:
            self.table_lines.pop(0).remove()
            
        # 确保至少有3个参考标签可见
        if len(reference_tags_poses) < 3:
            return
            
        # 计算参考标签的中心点坐标
        tag_positions = [pos for pos, _ in reference_tags_poses.values()]
        
        # 尝试通过凸包算法找到桌子边缘
        if len(tag_positions) >= 3:
            try:
                # 提取x,y坐标来形成2D平面
                points_2d = np.array([[p[0], p[1]] for p in tag_positions])
                
                # 计算2D凸包
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points_2d)
                
                # 绘制凸包边缘
                for simplex in hull.simplices:
                    i, j = simplex
                    line, = self.ax.plot3D(
                        [tag_positions[i][0], tag_positions[j][0]],
                        [tag_positions[i][1], tag_positions[j][1]],
                        [tag_positions[i][2], tag_positions[j][2]],
                        'gray', linewidth=1, alpha=0.5)
                    self.table_lines.append(line)
            except Exception as e:
                # 如果凸包计算失败，退回到简单连线
                print(f"构建桌面轮廓失败: {e}")
                self.draw_simple_table(tag_positions)
        else:
            self.draw_simple_table(tag_positions)
    
    def draw_simple_table(self, tag_positions):
        """简单地连接所有标签点"""
        # 按标签ID排序
        n = len(tag_positions)
        for i in range(n):
            j = (i + 1) % n
            line, = self.ax.plot3D(
                [tag_positions[i][0], tag_positions[j][0]],
                [tag_positions[i][1], tag_positions[j][1]],
                [tag_positions[i][2], tag_positions[j][2]],
                'gray', linewidth=1, alpha=0.5)
            self.table_lines.append(line)

    def update(self, tag_poses, moving_tag_id, reference_tags):
        """更新3D显示"""
        # 检查是否需要调整坐标范围
        self.update_axis_limits(tag_poses)
        if self.needs_axis_update:
            self.ax.set_xlim(self.x_min, self.x_max)
            self.ax.set_ylim(self.y_min, self.y_max) 
            self.ax.set_zlim(self.z_min, self.z_max)
            self.needs_axis_update = False
        
        # 清除旧图形
        while self.quivers:
            self.quivers.pop(0).remove()
        while self.texts:
            self.texts.pop(0).remove()
        while self.scatter_points:
            self.scatter_points.pop(0).remove()

        # 分离参考标签和移动标签
        reference_tags_poses = {id: pose for id, pose in tag_poses.items() 
                              if id in reference_tags}
        
        # 绘制桌面
        self.draw_table(reference_tags_poses)
        
        # 绘制每个标签
        for tag_id, (t, R) in tag_poses.items():
            if tag_id == moving_tag_id:  # 特殊处理移动标签
                color = 'magenta'
                axis_scale = 0.1
                scatter_size = 100
                
                # 记录移动标签轨迹
                self.track_history.append(t.copy())
                if len(self.track_history) > 100:  # 限制历史长度
                    self.track_history.pop(0)
                    
                # 更新轨迹线
                if self.track_line:
                    self.track_line.remove()
                if len(self.track_history) > 1:
                    track_array = np.array(self.track_history)
                    self.track_line, = self.ax.plot3D(
                        track_array[:, 0], track_array[:, 1], track_array[:, 2],
                        'red', linewidth=1, alpha=0.5)
            else:
                color = 'cyan'
                axis_scale = 0.05
                scatter_size = 70
                
            # 绘制位置点
            scatter = self.ax.scatter(t[0], t[1], t[2], c=color, s=scatter_size)
            self.scatter_points.append(scatter)

            # 绘制坐标系轴
            x_end = t + R[:,0] * axis_scale
            y_end = t + R[:,1] * axis_scale
            z_end = t + R[:,2] * axis_scale

            # X轴 (红色)
            self.quivers.append(self.ax.quiver(
                t[0], t[1], t[2],
                x_end[0]-t[0], x_end[1]-t[1], x_end[2]-t[2],
                color='red', linewidth=1.5))

            # Y轴 (绿色)
            self.quivers.append(self.ax.quiver(
                t[0], t[1], t[2],
                y_end[0]-t[0], y_end[1]-t[1], y_end[2]-t[2],
                color='green', linewidth=1.5))

            # Z轴 (蓝色)
            self.quivers.append(self.ax.quiver(
                t[0], t[1], t[2],
                z_end[0]-t[0], z_end[1]-t[1], z_end[2]-t[2],
                color='blue', linewidth=1.5))

            # 添加标签ID (使用英文)
            text = self.ax.text(t[0], t[1], t[2]+0.03, str(tag_id),
                              color='black', fontsize=10, ha='center')
            self.texts.append(text)

        # 刷新图形
        self.fig.canvas.draw()
        plt.pause(0.001)

# ================== 主程序 ==================
def main():
    parser = argparse.ArgumentParser(description='Desktop AprilTag Tracking System')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--config', default='config/vision/table_setup.json',
                       help='Config file path (default: config/vision/table_setup.json)')
    parser.add_argument('--camera_info', default='config/camera/camera_info_1.yaml',
                       help='Camera parameter file (default: config/camera/camera_info_1.yaml)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera width resolution (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera height resolution (default: 720)')
    args = parser.parse_args()

    # 初始化配置
    try:
        print(f"Reading config file: {args.config}")
        config = read_json(args.config)
        apriltag_config = AprilTagConfig(**config["AprilTagConfig"])
        archive_config = ArchiveConfig(**config["Archive"])
        table_config = TableConfig(**config["TableConfig"])
        
        print(f"Reading camera parameters: {args.camera_info}")
        K, D = read_camera_info(args.camera_info)
        print(f"Camera matrix K:\n{K}")
        print(f"Distortion coefficients D: {D}")
    except Exception as e:
        print(f"Config loading failed: {e}")
        print("Using default settings...")
        # 创建默认配置
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        D = np.zeros((4, 1), dtype=np.float32)
        
        # 使用默认的AprilTag配置
        class DefaultConfig:
            def __init__(self):
                self.family = "tag36h11"
                self.size = 0.1  # 10厘米
                self.threads = 2
                self.max_hamming = 0
                self.z_up = True
                self.decimate = 1.0
                self.blur = 0.0
                self.refine_edges = 1
                self.debug = 0
                
        apriltag_config = DefaultConfig()
        
        # 默认桌面配置
        class DefaultTableConfig:
            def __init__(self):
                self.reference_tags = [0, 1, 2, 3]
                self.moving_tag = 4
                self.tag_positions = {
                    0: [0.0, 0.0, 0.0],
                    1: [0.5, 0.0, 0.0],
                    2: [0.5, 0.5, 0.0],
                    3: [0.0, 0.5, 0.0]
                }
        
        table_config = DefaultTableConfig()
        
        # 创建默认归档配置
        class DefaultArchive:
            def __init__(self):
                self.enable = False
                self.preview = True
                self.save_raw = False
                self.save_pred = False
                self.preview_delay = 1
                self.path = "./data/table_tracking"
                
        archive_config = DefaultArchive()

    # 初始化可视化
    visualizer = TableVisualizer()

    # 初始化AprilTag检测器
    print("Initializing AprilTag detector...")
    detector = Detector(DetectorOptions(
        families=apriltag_config.family,
        nthreads=apriltag_config.threads,
        quad_blur=apriltag_config.blur,
        quad_decimate=apriltag_config.decimate,
        refine_edges=bool(apriltag_config.refine_edges),
        debug=bool(apriltag_config.debug)
    ))
    
    # 初始化相机
    print(f"Opening camera ID: {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        sys.exit(1)
    
    # 设置相机分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 创建存档目录
    if archive_config.enable:
        create_dirs_if_not_exist(archive_config.path)
    
    # 创建数据记录文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(
        archive_config.path if archive_config.enable else './data',
        f"tag_tracking_{timestamp}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w') as f:
        f.write("timestamp,tag_id,x,y,z,qx,qy,qz,qw\n")
    
    print("Starting AprilTag tracking...")
    print("Press 'q' to quit")
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("Could not read camera frame")
                break
            
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测AprilTags
            start_time = time.time()
            tags = detector.detect(gray)
            end_time = time.time()
            if (end_time - start_time) != 0:
                fps = 1.0 / (end_time - start_time)
            else:
                fps = 0
            
            # 存储所有标签的位姿
            tag_poses = {}
            
            # 处理检测结果
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
                
                # 存储此标签的位姿
                tag_poses[tag.tag_id] = (tvec, R)
                
                # 记录数据
                if archive_config.enable:
                    current_time = time.time()
                    with open(csv_path, 'a') as f:
                        f.write(f"{current_time},{tag.tag_id},{tvec[0]},{tvec[1]},{tvec[2]},{quat[0]},{quat[1]},{quat[2]},{quat[3]}\n")
                
                # 在图像上标注ID和位置（使用英文）
                center = np.mean(camera_apriltag_corners, axis=0).astype(int)
                cv2.putText(frame, f"ID:{tag.tag_id}", (center[0], center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 检查是否找到移动标签
            if table_config.moving_tag in tag_poses:
                moving_tag_pose = tag_poses[table_config.moving_tag]
                # 计算并显示距离信息（使用英文）
                position_text = f"Moving Tag (ID:{table_config.moving_tag}) Position: "
                position_text += f"X:{moving_tag_pose[0][0]:.2f} Y:{moving_tag_pose[0][1]:.2f} Z:{moving_tag_pose[0][2]:.2f}"
                cv2.putText(frame, position_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 在图像上绘制检测结果
            output_image = draw_detection_results(
                frame, 
                tags, 
                K, D, 
                apriltag_config.size,
                flip_z=apriltag_config.z_up
            )
            
            # 添加FPS显示
            cv2.putText(output_image, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 更新3D可视化
            visualizer.update(tag_poses, table_config.moving_tag, table_config.reference_tags)
            
            # 显示结果
            if archive_config.preview:
                cv2.imshow("AprilTag Table Tracking", output_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # 保存图像
            if archive_config.enable and archive_config.save_pred:
                frame_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                pred_image_name = os.path.join(archive_config.path, f"{frame_timestamp}_pred.jpg")
                cv2.imwrite(pred_image_name, output_image)
    
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        print("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        plt.close()
        print("Program ended")

if __name__ == "__main__":
    main()