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
from modules.reference import TableReference
from modules.kalman import TableKalmanFilter  # 导入卡尔曼滤波器

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
        
        # 初始化参考系统
        self.reference = TableReference(
            reference_tag_ids=table_config.reference_tags,
            moving_tag_ids=table_config.moving_tags
        )
        
        # 初始化卡尔曼滤波器
        self.kalman_filter = TableKalmanFilter(
            reference_tags=table_config.reference_tags,
            moving_tags=table_config.moving_tags,
            use_acceleration=True,  # 移动标签使用加速度模型
            ref_process_noise=1e-5,  # 参考标签过程噪声（较小，因为固定）
            ref_measure_noise=1e-3,  # 参考标签测量噪声
            moving_process_noise=1e-4,  # 移动标签过程噪声
            moving_measure_noise=1e-2  # 移动标签测量噪声
        )
        
        # 是否启用卡尔曼滤波
        self.use_kalman = True
        
        # 是否应用平面约束（固定标签位于同一平面）
        self.apply_plane_constraint = True
        
        # 上次处理时间
        self.last_timestamp = None
        
        # 初始化标志
        self.initialized = False
        
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
    
    def initialize_system(self, frame):
        """拍照初始化系统，为后续跟踪建立参考
        
        Args:
            frame: 输入的相机图像帧
            
        Returns:
            bool: 是否成功初始化
        """
        print("开始系统初始化...")
        
        # 获取当前帧标签位姿
        tag_poses = self.get_tag_poses(frame)
        
        # 应用平面约束（如果启用）
        if self.apply_plane_constraint:
            tag_poses = self._apply_plane_constraint(tag_poses)
        
        # 重置卡尔曼滤波器
        self.kalman_filter.reset()
        
        # 尝试初始化参考系统
        if self.reference.initialize(tag_poses):
            self.initialized = True
            self.table_config.initialized = True
            print("系统初始化成功！可以开始跟踪")
            return True
        else:
            print("系统初始化失败，请确保参考标签和移动标签在视野中")
            return False
    
    def _apply_plane_constraint(self, tag_poses):
        """应用平面约束，使所有参考标签的z坐标保持一致
        
        Args:
            tag_poses: 标签位姿字典 {tag_id: (position, rotation_matrix)}
            
        Returns:
            dict: 应用约束后的标签位姿字典
        """
        # 仅处理参考标签
        reference_tags = [tag_id for tag_id in tag_poses if tag_id in self.table_config.reference_tags]
        
        # 如果参考标签数量不足，无法应用约束
        if len(reference_tags) < 3:
            return tag_poses
        
        # 创建结果字典的副本
        constrained_poses = tag_poses.copy()
        
        # 收集参考标签位置
        ref_positions = np.array([tag_poses[tag_id][0] for tag_id in reference_tags])
        
        # 使用最小二乘法拟合平面 ax + by + c = z
        A = np.column_stack((ref_positions[:, 0], ref_positions[:, 1], np.ones(len(reference_tags))))
        b = ref_positions[:, 2]
        
        try:
            # 求解平面参数 [a, b, c]
            plane_params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # 对每个参考标签应用平面约束
            for tag_id in reference_tags:
                pos, rot = constrained_poses[tag_id]
                
                # 计算平面上的z坐标
                expected_z = plane_params[0] * pos[0] + plane_params[1] * pos[1] + plane_params[2]
                
                # 更新z坐标
                pos_constrained = pos.copy()
                pos_constrained[2] = expected_z
                
                # 调整旋转矩阵使z轴垂直于平面
                rot_constrained = self._align_rotation_to_plane(rot, plane_params)
                
                # 更新位姿
                constrained_poses[tag_id] = (pos_constrained, rot_constrained)
        except np.linalg.LinAlgError:
            # 如果最小二乘法失败，保持原样
            pass
        
        return constrained_poses
    
    def _align_rotation_to_plane(self, rotation_matrix, plane_params):
        """调整旋转矩阵使z轴垂直于平面
        
        Args:
            rotation_matrix: 原旋转矩阵
            plane_params: 平面参数 [a, b, c]
            
        Returns:
            numpy.ndarray: 调整后的旋转矩阵
        """
        # 计算平面法向量
        plane_normal = np.array([plane_params[0], plane_params[1], -1.0])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # 假设z轴应该与平面法向量对齐
        z_axis = plane_normal
        
        # 获取原旋转矩阵的x轴和y轴
        x_axis = rotation_matrix[:, 0]
        y_axis = rotation_matrix[:, 1]
        
        # 使x轴与z轴正交
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # 使y轴与x轴和z轴都正交
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 创建新的旋转矩阵
        aligned_rotation = np.column_stack((x_axis, y_axis, z_axis))
        
        return aligned_rotation
    
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
        # 记录当前时间戳
        current_timestamp = time.time()
        
        # 如果系统尚未初始化且需要初始化
        if not self.initialized and self.table_config.require_initialization:
            # 尝试初始化
            self.initialize_system(frame)
            self.last_timestamp = current_timestamp
        
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
            
            # 存储此标签的位姿
            tag_poses[tag.tag_id] = (tvec, R)
        
        # 应用平面约束（如果启用）
        if self.apply_plane_constraint:
            tag_poses = self._apply_plane_constraint(tag_poses)
        
        # 如果系统已初始化，处理标签遮挡
        if self.initialized:
            # 尝试更新移动标签之间的相对关系
            self.reference.update_moving_relations(tag_poses)
            
            # 估计被遮挡标签的位姿
            complete_tag_poses = self.reference.compute_missing_tags(tag_poses)
            
            # 标记哪些标签是估计的（在图像中没有检测到）
            estimated_tags = [tag_id for tag_id in complete_tag_poses.keys() if tag_id not in tag_poses]
            
            # 使用完整的位姿集
            tag_poses = complete_tag_poses
            
            # 添加估计标签的标记（在图像中没有，但估计了位置）
            for tag_id in estimated_tags:
                cv2.putText(output_image, f"Estimated ID:{tag_id}", (10, 90 + 30 * (tag_id - estimated_tags[0])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 应用卡尔曼滤波（如果启用）
        if self.use_kalman:
            # 计算时间差
            dt = current_timestamp - self.last_timestamp if self.last_timestamp is not None else 0
            self.last_timestamp = current_timestamp
            
            # 应用滤波
            if dt > 0:  # 只有在有效时间差的情况下才应用滤波
                tag_poses = self.kalman_filter.update(tag_poses, current_timestamp)
        
        # 记录数据
        if self.csv_path:
            for tag_id, (tvec, R) in tag_poses.items():
                quat = matrix_to_quaternion(R)
                with open(self.csv_path, 'a') as f:
                    f.write(f"{current_timestamp},{tag_id},{tvec[0]},{tvec[1]},{tvec[2]},{quat[0]},{quat[1]},{quat[2]},{quat[3]}\n")
        
        # 检查是否找到移动标签
        for idx, moving_tag_id in enumerate(self.table_config.moving_tags):
            if moving_tag_id in tag_poses:
                moving_tag_pose = tag_poses[moving_tag_id]
                # 计算并显示距离信息
                position_text = f"移动标签 (ID:{moving_tag_id}) 位置: "
                position_text += f"X:{moving_tag_pose[0][0]:.2f} Y:{moving_tag_pose[0][1]:.2f} Z:{moving_tag_pose[0][2]:.2f}"
                cv2.putText(output_image, position_text, (10, 30 + idx * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 在图像上标注ID和位置
        for tag in tags:
            center = np.mean(tag.corners, axis=0).astype(int)
            cv2.putText(output_image, f"ID:{tag.tag_id}", (center[0], center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 添加FPS显示
        cv2.putText(output_image, f"FPS: {fps:.1f}", (10, output_image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加系统状态信息
        status_text = "系统状态: " + ("已初始化" if self.initialized else "未初始化")
        status_text += " | 滤波: " + ("开启" if self.use_kalman else "关闭")
        status_text += " | 平面约束: " + ("开启" if self.apply_plane_constraint else "关闭")
        cv2.putText(output_image, status_text, (output_image.shape[1] - 400, 30),
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
                self.K,
                self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            # 如果需要翻转Z轴
            if self.apriltag_config.z_up:
                tvec[2] = -tvec[2]
                rvec[0] = -rvec[0]
                rvec[1] = -rvec[1]
            
            # 使用Rodrigues变换将旋转向量转换为旋转矩阵
            R, _ = cv2.Rodrigues(rvec)
            tvec = tvec.flatten()
            
            # 存储此标签的位姿
            tag_poses[tag.tag_id] = (tvec, R)
        
        # 应用平面约束（如果启用）
        if self.apply_plane_constraint:
            tag_poses = self._apply_plane_constraint(tag_poses)
            
        return tag_poses 