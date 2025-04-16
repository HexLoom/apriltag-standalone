"""
相机标定和操作模块

这个模块包含相机标定和操作相关的函数
"""

from lib.camera.calibration import save_camera_calibration, create_dirs_if_not_exist

__all__ = ['save_camera_calibration', 'create_dirs_if_not_exist'] 