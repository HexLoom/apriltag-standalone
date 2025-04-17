#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置模块 - 定义了配置类和配置加载函数
"""

import json
import cv2
import numpy as np

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

def load_config(config_path, camera_info_path):
    """加载配置文件和相机参数"""
    try:
        print(f"读取配置文件: {config_path}")
        config = read_json(config_path)
        apriltag_config = AprilTagConfig(**config["AprilTagConfig"])
        archive_config = ArchiveConfig(**config["Archive"])
        table_config = TableConfig(**config["TableConfig"])
        
        print(f"读取相机参数: {camera_info_path}")
        K, D = read_camera_info(camera_info_path)
        print(f"相机矩阵 K:\n{K}")
        print(f"畸变系数 D: {D}")
        
        return apriltag_config, archive_config, table_config, K, D
    except Exception as e:
        print(f"配置加载失败: {e}")
        print("使用默认设置...")
        return create_default_configs()

def create_default_configs():
    """创建默认配置"""
    # 默认相机矩阵和畸变系数
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    D = np.zeros((4, 1), dtype=np.float32)
    
    # 默认AprilTag配置
    apriltag_config = AprilTagConfig(
        family="tag36h11",
        size=0.1,  # 10厘米
        threads=2,
        max_hamming=0,
        z_up=True,
        decimate=1.0,
        blur=0.0,
        refine_edges=1,
        debug=0
    )
    
    # 默认桌面配置
    table_config = TableConfig(
        reference_tags=[0, 1, 2, 3],
        moving_tag=4,
        tag_positions={
            0: [0.0, 0.0, 0.0],
            1: [0.5, 0.0, 0.0],
            2: [0.5, 0.5, 0.0],
            3: [0.0, 0.5, 0.0]
        }
    )
    
    # 默认归档配置
    archive_config = ArchiveConfig(
        enable=False,
        preview=True,
        save_raw=False,
        save_pred=False,
        preview_delay=1,
        path="./data/table_tracking"
    )
    
    return apriltag_config, archive_config, table_config, K, D 