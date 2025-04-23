#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置模块 - 定义了配置类和配置加载函数
"""

import json
import cv2
import numpy as np
import yaml

def read_json(file_path):
    """读取JSON配置文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def read_camera_info(yaml_path):
    """从OpenCV YAML文件读取相机参数"""
    try:
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
    except Exception as e:
        print(f"读取相机参数文件失败: {e}")
        raise

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

class CameraConfig:
    def __init__(self, device_id=0, width=1280, height=720, camera_info_path="config/camera/HSK_200W53_1080P.yaml",
                undistort=True, keep_fov=True, alpha=0.85):
        self.device_id = device_id           # 相机设备ID
        self.width = width                   # 相机宽度
        self.height = height                 # 相机高度
        self.camera_info_path = camera_info_path  # 相机参数文件路径
        self.undistort = undistort           # 是否校正畸变
        self.keep_fov = keep_fov             # 是否保持视场角
        self.alpha = alpha                   # 视场保留比例

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
    def __init__(self, reference_tags, moving_tag=None, moving_tags=None, tag_positions=None, require_initialization=True):
        self.reference_tags = reference_tags  # 参考标签ID列表
        
        # 兼容旧配置（单个移动标签）和新配置（多个移动标签）
        if moving_tags is not None:
            self.moving_tags = moving_tags  # 多个移动标签ID列表
        elif moving_tag is not None:
            self.moving_tags = [moving_tag]  # 转换单个移动标签为列表
            self.moving_tag = moving_tag     # 保留向后兼容性
        else:
            self.moving_tags = []
            self.moving_tag = None
            
        # 参考标签预设位置
        self.tag_positions = {}
        
        # 将字符串键转为整数
        if tag_positions:
            for tag_id, position in tag_positions.items():
                self.tag_positions[int(tag_id)] = position
                
        # 初始化状态
        self.initialized = False
        
        # 是否需要执行标签初始化
        self.require_initialization = require_initialization

def load_config(config_path):
    """加载配置文件和相机参数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        tuple: (apriltag_config, camera_config, archive_config, table_config, K, D)
    """
    try:
        print(f"读取配置文件: {config_path}")
        config = read_json(config_path)
        
        # 加载AprilTag配置
        apriltag_config = AprilTagConfig(**config["AprilTagConfig"])
        
        # 加载相机配置（确保处理未定义的参数）
        camera_dict = config.get("Camera", {})
        # 确保校正参数存在，使用默认值
        if "undistort" not in camera_dict:
            camera_dict["undistort"] = True
        if "keep_fov" not in camera_dict:
            camera_dict["keep_fov"] = True
        if "alpha" not in camera_dict:
            camera_dict["alpha"] = 0.85
            
        camera_config = CameraConfig(**camera_dict)
        
        # 打印相机校正信息
        print(f"相机畸变校正: {'开启' if camera_config.undistort else '关闭'}")
        if camera_config.undistort:
            print(f"  - 视场保持: {'开启' if camera_config.keep_fov else '关闭'}")
            print(f"  - 视场比例 alpha: {camera_config.alpha}")
        
        # 加载存档配置
        archive_config = ArchiveConfig(**config["Archive"])
        
        # 加载桌面配置
        table_config = TableConfig(**config["TableConfig"])
        
        # 加载相机参数
        print(f"读取相机参数: {camera_config.camera_info_path}")
        K, D = read_camera_info(camera_config.camera_info_path)
        print(f"相机矩阵 K:\n{K}")
        print(f"畸变系数 D: {D}")
        
        return apriltag_config, camera_config, archive_config, table_config, K, D
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
    
    # 默认相机配置
    camera_config = CameraConfig(
        device_id=0,
        width=1280,
        height=720,
        camera_info_path="config/camera/HSK_200W53_1080P.yaml",
        undistort=True,        # 默认开启畸变校正
        keep_fov=True,         # 默认保持视场角
        alpha=0.85            # 默认视场比例
    )
    
    # 默认桌面配置
    table_config = TableConfig(
        reference_tags=[0, 1, 2, 3],
        moving_tags=[4, 5, 6],  # 默认支持多个移动标签
        require_initialization=True,  # 默认需要初始化
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
    
    return apriltag_config, camera_config, archive_config, table_config, K, D 