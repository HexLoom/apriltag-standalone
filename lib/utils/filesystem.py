#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件系统工具模块
处理目录创建等操作
"""

import os

def create_dirs_if_not_exist(path):
    """
    如果目录不存在则创建
    
    参数:
        path: 目录路径
    """
    if not os.path.exists(path):
        print(f"创建目录: {path}")
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"创建目录错误: {e}") 