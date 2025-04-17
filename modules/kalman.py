#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
卡尔曼滤波器模块 - 用于标签位姿的平滑滤波
"""

import numpy as np
import cv2

class TagKalmanFilter:
    """标签位姿的卡尔曼滤波器"""
    
    def __init__(self, use_acceleration=False, process_noise=1e-5, measure_noise=1e-2, initial_pose=None):
        """初始化卡尔曼滤波器
        
        Args:
            use_acceleration: 是否使用加速度模型（如果为False，使用恒速模型）
            process_noise: 过程噪声系数
            measure_noise: 测量噪声系数
            initial_pose: 初始位姿元组 (position, rotation_matrix)
        """
        self.use_acceleration = use_acceleration
        self.initialized = False
        self.last_time = None
        
        # 位置状态维度 (x, y, z, vx, vy, vz) 或 (x, y, z, vx, vy, vz, ax, ay, az)
        self.pos_state_dim = 9 if use_acceleration else 6
        self.pos_measure_dim = 3  # 位置测量维度 (x, y, z)
        
        # 旋转状态维度（使用四元数表示 qw, qx, qy, qz 及其速度）
        self.rot_state_dim = 8
        self.rot_measure_dim = 4  # 旋转测量维度 (qw, qx, qy, qz)
        
        # 位置卡尔曼滤波器
        self.pos_kalman = cv2.KalmanFilter(self.pos_state_dim, self.pos_measure_dim)
        
        # 旋转卡尔曼滤波器
        self.rot_kalman = cv2.KalmanFilter(self.rot_state_dim, self.rot_measure_dim)
        
        # 设置参数
        self.setup_kalman_parameters(process_noise, measure_noise)
        
        # 设置初始状态
        if initial_pose is not None:
            self.initialize(initial_pose)
    
    def setup_kalman_parameters(self, process_noise, measure_noise):
        """设置卡尔曼滤波器参数"""
        # 位置卡尔曼滤波器参数
        
        # 状态转移矩阵 A
        self.pos_kalman.transitionMatrix = np.eye(self.pos_state_dim, dtype=np.float32)
        
        # 测量矩阵 H
        self.pos_kalman.measurementMatrix = np.zeros((self.pos_measure_dim, self.pos_state_dim), dtype=np.float32)
        self.pos_kalman.measurementMatrix[:, :self.pos_measure_dim] = np.eye(self.pos_measure_dim)
        
        # 过程噪声协方差矩阵 Q
        self.pos_kalman.processNoiseCov = np.eye(self.pos_state_dim, dtype=np.float32) * process_noise
        
        # 测量噪声协方差矩阵 R
        self.pos_kalman.measurementNoiseCov = np.eye(self.pos_measure_dim, dtype=np.float32) * measure_noise
        
        # 旋转卡尔曼滤波器参数
        
        # 状态转移矩阵 A
        self.rot_kalman.transitionMatrix = np.eye(self.rot_state_dim, dtype=np.float32)
        
        # 测量矩阵 H
        self.rot_kalman.measurementMatrix = np.zeros((self.rot_measure_dim, self.rot_state_dim), dtype=np.float32)
        self.rot_kalman.measurementMatrix[:, :self.rot_measure_dim] = np.eye(self.rot_measure_dim)
        
        # 过程噪声协方差矩阵 Q
        self.rot_kalman.processNoiseCov = np.eye(self.rot_state_dim, dtype=np.float32) * process_noise
        
        # 测量噪声协方差矩阵 R
        self.rot_kalman.measurementNoiseCov = np.eye(self.rot_measure_dim, dtype=np.float32) * measure_noise
    
    def initialize(self, pose, timestamp=None):
        """使用初始位姿初始化滤波器
        
        Args:
            pose: 位姿元组 (position, rotation_matrix)
            timestamp: 时间戳，如果不提供则使用当前时间
        """
        position, rotation = pose
        
        # 将旋转矩阵转换为四元数
        quat = self._rotation_matrix_to_quaternion(rotation)
        
        # 初始化位置滤波器状态
        pos_state = np.zeros((self.pos_state_dim, 1), dtype=np.float32)
        pos_state[:3, 0] = position  # x, y, z
        self.pos_kalman.statePost = pos_state
        self.pos_kalman.statePre = pos_state
        
        # 初始化旋转滤波器状态
        rot_state = np.zeros((self.rot_state_dim, 1), dtype=np.float32)
        rot_state[:4, 0] = quat  # qw, qx, qy, qz
        self.rot_kalman.statePost = rot_state
        self.rot_kalman.statePre = rot_state
        
        # 记录时间戳
        self.last_time = timestamp if timestamp is not None else 0
        
        # 标记为已初始化
        self.initialized = True
    
    def update(self, pose, timestamp=None):
        """使用新的观测值更新滤波器，并返回滤波后的结果
        
        Args:
            pose: 位姿元组 (position, rotation_matrix)
            timestamp: 时间戳，如果不提供则默认dt=1
            
        Returns:
            tuple: 滤波后的位姿 (position, rotation_matrix)
        """
        if not self.initialized:
            self.initialize(pose, timestamp)
            return pose
        
        # 计算时间差
        current_time = timestamp if timestamp is not None else self.last_time + 1
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # 仅当dt有效时更新状态转移矩阵
        if dt > 0:
            self._update_transition_matrix(dt)
        
        position, rotation = pose
        
        # 将旋转矩阵转换为四元数
        quat = self._rotation_matrix_to_quaternion(rotation)
        
        # 检查四元数连续性，避免突变
        self._ensure_quaternion_continuity(quat)
        
        # 预测状态
        self.pos_kalman.predict()
        self.rot_kalman.predict()
        
        # 准备测量值
        pos_measure = np.array(position.reshape(self.pos_measure_dim, 1), dtype=np.float32)
        rot_measure = np.array(quat.reshape(self.rot_measure_dim, 1), dtype=np.float32)
        
        # 更新滤波器
        filtered_pos = self.pos_kalman.correct(pos_measure)
        filtered_rot = self.rot_kalman.correct(rot_measure)
        
        # 从滤波状态提取位置和旋转
        filtered_position = filtered_pos[:3, 0]
        filtered_quaternion = filtered_rot[:4, 0]
        
        # 将四元数标准化并转换回旋转矩阵
        filtered_quaternion = self._normalize_quaternion(filtered_quaternion)
        filtered_rotation = self._quaternion_to_rotation_matrix(filtered_quaternion)
        
        return (filtered_position, filtered_rotation)
    
    def _update_transition_matrix(self, dt):
        """根据时间间隔更新状态转移矩阵
        
        Args:
            dt: 时间间隔
        """
        # 更新位置状态转移矩阵
        for i in range(3):  # 对x, y, z三个坐标
            # 位置 + 速度*dt
            self.pos_kalman.transitionMatrix[i, i+3] = dt
            
            if self.use_acceleration and i+6 < self.pos_state_dim:
                # 速度 + 加速度*dt
                self.pos_kalman.transitionMatrix[i+3, i+6] = dt
                # 位置 + 0.5*加速度*dt^2
                self.pos_kalman.transitionMatrix[i, i+6] = 0.5 * dt * dt
        
        # 更新旋转状态转移矩阵
        for i in range(4):  # 对qw, qx, qy, qz四个分量
            # 旋转 + 角速度*dt
            self.rot_kalman.transitionMatrix[i, i+4] = dt
    
    def _rotation_matrix_to_quaternion(self, rotation_matrix):
        """将旋转矩阵转换为四元数
        
        Args:
            rotation_matrix: 3x3旋转矩阵
            
        Returns:
            numpy.ndarray: 四元数 [qw, qx, qy, qz]
        """
        trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            S = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qx = 0.25 * S
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            S = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qy = 0.25 * S
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
        else:
            S = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
            qz = 0.25 * S
            
        # 返回 [qw, qx, qy, qz] 形式的四元数
        return np.array([qw, qx, qy, qz], dtype=np.float32)
    
    def _quaternion_to_rotation_matrix(self, quaternion):
        """将四元数转换为旋转矩阵
        
        Args:
            quaternion: 四元数 [qw, qx, qy, qz]
            
        Returns:
            numpy.ndarray: 3x3旋转矩阵
        """
        qw, qx, qy, qz = quaternion
        
        # 计算旋转矩阵
        rotation_matrix = np.zeros((3, 3), dtype=np.float32)
        
        # 四元数转换为旋转矩阵的公式
        rotation_matrix[0, 0] = 1 - 2*qy*qy - 2*qz*qz
        rotation_matrix[0, 1] = 2*qx*qy - 2*qz*qw
        rotation_matrix[0, 2] = 2*qx*qz + 2*qy*qw
        
        rotation_matrix[1, 0] = 2*qx*qy + 2*qz*qw
        rotation_matrix[1, 1] = 1 - 2*qx*qx - 2*qz*qz
        rotation_matrix[1, 2] = 2*qy*qz - 2*qx*qw
        
        rotation_matrix[2, 0] = 2*qx*qz - 2*qy*qw
        rotation_matrix[2, 1] = 2*qy*qz + 2*qx*qw
        rotation_matrix[2, 2] = 1 - 2*qx*qx - 2*qy*qy
        
        return rotation_matrix
    
    def _normalize_quaternion(self, quaternion):
        """标准化四元数
        
        Args:
            quaternion: 四元数 [qw, qx, qy, qz]
            
        Returns:
            numpy.ndarray: 标准化后的四元数
        """
        norm = np.linalg.norm(quaternion)
        if norm > 0:
            return quaternion / norm
        return quaternion
    
    def _ensure_quaternion_continuity(self, quaternion):
        """确保四元数连续性，避免突变
        
        Args:
            quaternion: 当前四元数 [qw, qx, qy, qz]
        """
        if not self.initialized:
            return
        
        # 获取上一个四元数
        prev_quaternion = self.rot_kalman.statePost[:4, 0]
        
        # 计算点积
        dot_product = np.sum(quaternion * prev_quaternion)
        
        # 如果点积为负，取反四元数
        if dot_product < 0:
            quaternion *= -1


class TableKalmanFilter:
    """桌面标签跟踪的卡尔曼滤波器管理器"""
    
    def __init__(self, reference_tags, moving_tags, use_acceleration=False, 
                 ref_process_noise=1e-5, ref_measure_noise=1e-3,
                 moving_process_noise=1e-4, moving_measure_noise=1e-2):
        """初始化桌面标签卡尔曼滤波器管理器
        
        Args:
            reference_tags: 参考标签ID列表
            moving_tags: 移动标签ID列表
            use_acceleration: 是否使用加速度模型
            ref_process_noise: 参考标签过程噪声
            ref_measure_noise: 参考标签测量噪声
            moving_process_noise: 移动标签过程噪声
            moving_measure_noise: 移动标签测量噪声
        """
        self.reference_tags = reference_tags
        self.moving_tags = moving_tags
        
        # 创建滤波器字典
        self.filters = {}
        
        # 为参考标签和移动标签创建不同参数的滤波器
        for tag_id in reference_tags:
            self.filters[tag_id] = TagKalmanFilter(
                use_acceleration=False,  # 参考标签使用恒速模型
                process_noise=ref_process_noise,
                measure_noise=ref_measure_noise
            )
            
        for tag_id in moving_tags:
            self.filters[tag_id] = TagKalmanFilter(
                use_acceleration=use_acceleration,  # 移动标签可选加速度模型
                process_noise=moving_process_noise,
                measure_noise=moving_measure_noise
            )
    
    def update(self, tag_poses, timestamp=None):
        """更新所有可见标签的滤波器并返回滤波后的位姿
        
        Args:
            tag_poses: 标签位姿字典 {tag_id: (position, rotation_matrix)}
            timestamp: 时间戳
            
        Returns:
            dict: 滤波后的标签位姿字典
        """
        filtered_poses = {}
        
        # 对每个标签应用滤波
        for tag_id, pose in tag_poses.items():
            # 如果标签没有对应的滤波器，创建一个
            if tag_id not in self.filters:
                if tag_id in self.reference_tags:
                    self.filters[tag_id] = TagKalmanFilter(
                        use_acceleration=False,
                        process_noise=1e-5, # 参考标签过程噪声 
                        measure_noise=1e-3, # 参考标签测量噪声 增大值会更信任新测量，减少滞后
                        initial_pose=pose
                    )
                else:
                    self.filters[tag_id] = TagKalmanFilter(
                        use_acceleration=True,
                        process_noise=1e-1, # 移动标签过程噪声
                        measure_noise=1e-1, # 移动标签测量噪声
                        initial_pose=pose
                    )
            
            # 应用滤波
            filtered_pose = self.filters[tag_id].update(pose, timestamp)
            filtered_poses[tag_id] = filtered_pose
            
        return filtered_poses
    
    def reset(self):
        """重置所有滤波器"""
        for filter in self.filters.values():
            filter.initialized = False 