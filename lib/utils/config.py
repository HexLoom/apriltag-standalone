#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理模块
处理AprilTag和图像存档配置
"""

from dataclasses import dataclass
import json
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class AprilTagConfig:
    """AprilTag检测器配置类"""
    family: str
    size: float
    threads: int
    max_hamming: int
    z_up: bool
    decimate: float
    blur: float
    refine_edges: int
    debug: int

@dataclass
class ArchiveConfig:
    """图像存档配置类"""
    enable: bool
    preview: bool
    save_raw: bool
    save_pred: bool
    preview_delay: int
    path: str

def matrix_to_quaternion(rot_matrix):
    """
    将3x3旋转矩阵转换为四元数(x,y,z,w)
    
    参数:
        rot_matrix: 3x3旋转矩阵
        
    返回:
        四元数(x,y,z,w)数组
    """
    # 检查旋转矩阵的形状
    if rot_matrix.shape != (3, 3):
        raise ValueError("旋转矩阵必须是3x3")

    try:
        r = Rotation.from_matrix(rot_matrix)
        # 顺序 (x, y, z, w)
        quat = r.as_quat()
    except Exception as e:
        # 如果转换失败，可能是由于数值不稳定等问题导致
        print(f"四元数转换错误: {e}")
        return None
    return quat

def read_json(json_path):
    """
    读取JSON配置文件
    
    参数:
        json_path: JSON文件路径
        
    返回:
        解析后的JSON内容
    """
    with open(json_path, "r") as f:
        # 处理json文件可能包含注释的情况
        content = f.read()
        content = '\n'.join([line.split('//')[0] for line in content.split('\n')])
        data = json.loads(content)
    return data

def read_camera_info(yaml_path):
    """
    读取相机参数文件
    提取相机内参矩阵和畸变系数
    
    参数:
        yaml_path: 相机参数YAML文件路径
        
    返回:
        (K, D): 内参矩阵和畸变系数
    """
    import yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 提取相机内参矩阵K
    if 'camera_matrix' in data:
        K_data = data['camera_matrix']['data']
        K = np.array(K_data).reshape(3, 3)
    else:
        raise ValueError("相机参数文件中找不到camera_matrix")
    
    # 提取畸变系数D
    if 'distortion_coefficients' in data:
        D = np.array(data['distortion_coefficients']['data'])
    else:
        raise ValueError("相机参数文件中找不到distortion_coefficients")
        
    return K, D 