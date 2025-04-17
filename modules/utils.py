#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工具函数模块 - 提供通用的辅助函数
"""

import os
import numpy as np

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