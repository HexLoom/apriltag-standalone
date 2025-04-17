#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
相机标定实用工具
提供相机标定相关的函数
"""

import os
import numpy as np
import yaml
from filesystem import create_dirs_if_not_exist

def save_camera_calibration(output_file, camera_matrix, dist_coefs, image_width, image_height):
    """
    将相机标定结果保存为YAML文件
    
    参数:
        output_file: 输出文件路径
        camera_matrix: 相机内参矩阵
        dist_coefs: 畸变系数
        image_width: 图像宽度
        image_height: 图像高度
    """
    # 创建标定结果字典
    calibration_data = {
        'image_width': image_width,
        'image_height': image_height,
        'camera_name': 'usb_camera',
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': camera_matrix.flatten().tolist()
        },
        'distortion_model': 'plumb_bob',
        'distortion_coefficients': {
            'rows': 1,
            'cols': len(dist_coefs[0]),
            'data': dist_coefs.flatten().tolist()
        },
        'rectification_matrix': {
            'rows': 3,
            'cols': 3,
            'data': np.eye(3).flatten().tolist()
        },
        'projection_matrix': {
            'rows': 3,
            'cols': 4,
            'data': np.hstack((camera_matrix, np.zeros((3, 1)))).flatten().tolist()
        }
    }
    
    # 保存为YAML文件
    create_dirs_if_not_exist(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)
    
    print(f"标定结果已保存至 {output_file}") 