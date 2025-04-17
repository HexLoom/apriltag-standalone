#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
跟踪处理模块 - 处理AprilTag标签跟踪和位姿计算
"""

import cv2
import numpy as np
import time
from datetime import datetime
import os
from modules.utils import matrix_to_quaternion, create_dirs_if_not_exist

class TableTracker:
    def __init__(self, detector, camera_matrix, dist_coeffs, apriltag_config, table_config, archive_config=None):
        """初始化桌面跟踪器
        
        Args:
            detector: AprilTag检测器
            camera_matrix: 相机内参矩阵
            dist_coeffs: 相机畸变系数
            apriltag_config: AprilTag配置
            table_config: 桌面配置
            archive_config: 存档配置（可选）
        """
        self.detector = detector
        self.K = camera_matrix
        self.D = dist_coeffs
        self.apriltag_config = apriltag_config
        self.table_config = table_config
        self.archive_config = archive_config
        
        # 数据记录文件
        if archive_config and archive_config.enable:
            # 创建存档目录
            create_dirs_if_not_exist(archive_config.path)
            
            # 创建CSV文件
            self.csv_path = os.path.join(
                archive_config.path,
                f"tag_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            with open(self.csv_path, 'w') as f:
                f.write("timestamp,tag_id,x,y,z,qx,qy,qz,qw\n")
        else:
            self.csv_path = None
    
    def process_frame(self, frame):
        """处理一帧图像，返回处理结果
        
        Args:
            frame: 输入的相机图像帧
            
        Returns:
            tag_poses: 标签位姿字典 {tag_id: (position, rotation_matrix)}
            tags: 检测到的AprilTag标签列表
            output_image: 可视化结果图像
            fps: 处理帧率
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测AprilTags
        start_time = time.time()
        tags = self.detector.detect(gray)
        end_time = time.time()
        if (end_time - start_time) != 0:
            fps = 1.0 / (end_time - start_time)
        else:
            fps = 0
        
        # 存储所有标签的位姿
        tag_poses = {}
        
        # 制作输出图像的副本
        output_image = frame.copy()
        
        # 处理检测结果
        for tag in tags:
            # 提取角点
            camera_apriltag_corners = np.array(tag.corners, dtype=np.float32)
            
            # 世界坐标系中的角点 (标签坐标系)
            world_apriltag_corners = np.array([
                [-self.apriltag_config.size/2, self.apriltag_config.size/2, 0],
                [self.apriltag_config.size/2, self.apriltag_config.size/2, 0],
                [self.apriltag_config.size/2, -self.apriltag_config.size/2, 0],
                [-self.apriltag_config.size/2, -self.apriltag_config.size/2, 0]
            ], dtype=np.float32)
            
            # 使用solvePnP计算位姿
            # rvec 3x1 Rodrigues旋转向量
            # tvec 3x1 平移向量
            _, rvec, tvec = cv2.solvePnP(
                world_apriltag_corners,        # 世界坐标中的角点（3D）
                camera_apriltag_corners,       # 图像像素坐标中的角点（2D）
                self.K,                        # 相机内参矩阵
                self.D,                        # 畸变系数
                flags=cv2.SOLVEPNP_IPPE_SQUARE # 适合平面正方形的PnP方法
            )
            
            # 如果需要翻转Z轴
            if self.apriltag_config.z_up:
                tvec[2] = -tvec[2]
                # 旋转向量也需要调整
                rvec[0] = -rvec[0]
                rvec[1] = -rvec[1]
            
            # 使用Rodrigues变换将旋转向量转换为旋转矩阵
            R, _ = cv2.Rodrigues(rvec)
            tvec = tvec.flatten() # 将tvec展平为一维数组
            quat = matrix_to_quaternion(R)
            
            # 存储此标签的位姿
            tag_poses[tag.tag_id] = (tvec, R)
            
            # 记录数据
            if self.csv_path:
                current_time = time.time()
                with open(self.csv_path, 'a') as f:
                    f.write(f"{current_time},{tag.tag_id},{tvec[0]},{tvec[1]},{tvec[2]},{quat[0]},{quat[1]},{quat[2]},{quat[3]}\n")
            
            # 在图像上标注ID和位置
            center = np.mean(camera_apriltag_corners, axis=0).astype(int)
            cv2.putText(output_image, f"ID:{tag.tag_id}", (center[0], center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 检查是否找到移动标签
        if self.table_config.moving_tag in tag_poses:
            moving_tag_pose = tag_poses[self.table_config.moving_tag]
            # 计算并显示距离信息
            position_text = f"Mobile tag (ID:{self.table_config.moving_tag}) location: "
            position_text += f"X:{moving_tag_pose[0][0]:.2f} Y:{moving_tag_pose[0][1]:.2f} Z:{moving_tag_pose[0][2]:.2f}"
            cv2.putText(output_image, position_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加FPS显示
        cv2.putText(output_image, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存图像
        if self.archive_config and self.archive_config.enable:
            if self.archive_config.save_pred:
                frame_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                pred_image_name = os.path.join(self.archive_config.path, f"{frame_timestamp}_pred.jpg")
                cv2.imwrite(pred_image_name, output_image)
            
            if self.archive_config.save_raw:
                frame_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                raw_image_name = os.path.join(self.archive_config.path, f"{frame_timestamp}_raw.jpg")
                cv2.imwrite(raw_image_name, frame)
        
        return tag_poses, tags, output_image, fps
        
    def get_tag_poses(self, frame):
        """只处理标签位姿，不生成可视化结果
        
        Args:
            frame: 输入的相机图像帧
            
        Returns:
            tag_poses: 标签位姿字典 {tag_id: (position, rotation_matrix)}
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测AprilTags
        tags = self.detector.detect(gray)
        
        # 存储所有标签的位姿
        tag_poses = {}
        
        # 处理检测结果
        for tag in tags:
            # 提取角点
            camera_apriltag_corners = np.array(tag.corners, dtype=np.float32)
            
            # 世界坐标系中的角点 (标签坐标系)
            world_apriltag_corners = np.array([
                [-self.apriltag_config.size/2, self.apriltag_config.size/2, 0],
                [self.apriltag_config.size/2, self.apriltag_config.size/2, 0],
                [self.apriltag_config.size/2, -self.apriltag_config.size/2, 0],
                [-self.apriltag_config.size/2, -self.apriltag_config.size/2, 0]
            ], dtype=np.float32)
            
            # 使用solvePnP计算位姿
            _, rvec, tvec = cv2.solvePnP(
                world_apriltag_corners, 
                camera_apriltag_corners, 
                self.K, self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            # 如果需要翻转Z轴
            if self.apriltag_config.z_up:
                tvec[2] = -tvec[2]
                # 旋转向量也需要调整
                rvec[0] = -rvec[0]
                rvec[1] = -rvec[1]
            
            # 将旋转向量转换为旋转矩阵
            R, _ = cv2.Rodrigues(rvec)
            tvec = tvec.flatten()
            
            # 存储此标签的位姿
            tag_poses[tag.tag_id] = (tvec, R)
            
            # 记录数据
            if self.csv_path:
                quat = matrix_to_quaternion(R)
                current_time = time.time()
                with open(self.csv_path, 'a') as f:
                    f.write(f"{current_time},{tag.tag_id},{tvec[0]},{tvec[1]},{tvec[2]},{quat[0]},{quat[1]},{quat[2]},{quat[3]}\n")
        
        return tag_poses 