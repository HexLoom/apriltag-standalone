#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
桌面参考系统模块 - 记录标签的初始位姿并处理遮挡情况
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

class TableReference:
    def __init__(self, reference_tag_ids, moving_tag_ids):
        """初始化桌面参考系统
        
        Args:
            reference_tag_ids: 固定参考标签的ID列表
            moving_tag_ids: 移动标签的ID列表或单个ID
        """
        # 将单个移动标签转换为列表
        if isinstance(moving_tag_ids, int):
            moving_tag_ids = [moving_tag_ids]
            
        self.reference_tag_ids = reference_tag_ids
        self.moving_tag_ids = moving_tag_ids
        
        # 存储参考标签的初始位姿
        self.reference_poses = {}
        
        # 存储移动标签组的相对位姿关系
        self.moving_tag_relations = {}
        
        # 初始化状态
        self.initialized = False
        
    def initialize(self, tag_poses):
        """初始化参考系统，记录所有标签的初始位姿
        
        Args:
            tag_poses: 标签位姿字典 {tag_id: (position, rotation_matrix)}
            
        Returns:
            bool: 是否成功初始化
        """
        # 检查是否有足够的参考标签可见
        visible_refs = [tag_id for tag_id in self.reference_tag_ids if tag_id in tag_poses]
        if len(visible_refs) < 3:
            print(f"初始化失败：参考标签不足，需要至少3个，当前可见 {len(visible_refs)} 个")
            return False
            
        # 检查是否有足够的移动标签可见
        visible_movings = [tag_id for tag_id in self.moving_tag_ids if tag_id in tag_poses]
        if len(visible_movings) < 1:
            print(f"初始化失败：移动标签不可见")
            return False
            
        # 记录参考标签的初始位姿
        for tag_id in visible_refs:
            self.reference_poses[tag_id] = tag_poses[tag_id]
            
        # 记录移动标签之间的相对位姿关系
        if len(visible_movings) > 1:
            # 选择第一个移动标签作为组参考
            base_tag_id = visible_movings[0]
            base_pos, base_rot = tag_poses[base_tag_id]
            
            # 计算其他移动标签相对于基准移动标签的位姿
            for tag_id in visible_movings[1:]:
                curr_pos, curr_rot = tag_poses[tag_id]
                
                # 计算相对位置
                rel_pos = curr_pos - base_pos
                
                # 计算相对旋转（从基准标签到当前标签）
                rel_rot = np.dot(curr_rot, base_rot.T)
                
                # 存储相对位姿
                self.moving_tag_relations[tag_id] = (rel_pos, rel_rot)
                
        # 设置初始化完成标志
        self.initialized = True
        print(f"参考系统初始化成功！可见固定标签：{visible_refs}，可见移动标签：{visible_movings}")
        return True
        
    def compute_missing_tags(self, tag_poses):
        """计算被遮挡标签的估计位姿
        
        Args:
            tag_poses: 当前帧检测到的标签位姿
            
        Returns:
            dict: 包含所有可见和估计位姿的完整标签位姿字典
        """
        if not self.initialized:
            return tag_poses
            
        # 创建结果字典的副本
        result_poses = tag_poses.copy()
        
        # 处理缺失的参考标签
        visible_refs = [tag_id for tag_id in self.reference_tag_ids if tag_id in tag_poses]
        
        # 如果至少有两个参考标签可见，可以尝试估计缺失的参考标签
        if len(visible_refs) >= 2:
            for tag_id in self.reference_tag_ids:
                if tag_id not in tag_poses and tag_id in self.reference_poses:
                    # 使用初始位姿作为估计值
                    result_poses[tag_id] = self.reference_poses[tag_id]
        
        # 处理缺失的移动标签
        visible_movings = [tag_id for tag_id in self.moving_tag_ids if tag_id in tag_poses]
        
        # 如果至少有一个移动标签可见，并且有移动标签的相对关系记录
        if len(visible_movings) >= 1 and self.moving_tag_relations:
            # 以第一个可见的移动标签为基准
            base_tag_id = visible_movings[0]
            base_pos, base_rot = tag_poses[base_tag_id]
            
            # 估计其他缺失的移动标签
            for tag_id, (rel_pos, rel_rot) in self.moving_tag_relations.items():
                if tag_id not in tag_poses:
                    # 使用基准标签的当前位姿和相对位姿计算缺失标签的估计位姿
                    est_rot = np.dot(rel_rot, base_rot)
                    est_pos = base_pos + np.dot(base_rot, rel_pos)
                    
                    # 添加到结果中
                    result_poses[tag_id] = (est_pos, est_rot)
        
        return result_poses
        
    def update_moving_relations(self, tag_poses):
        """更新移动标签之间的相对位姿关系
        
        Args:
            tag_poses: 当前帧检测到的标签位姿
            
        Returns:
            bool: 是否成功更新
        """
        # 检查是否已初始化
        if not self.initialized:
            return False
            
        # 获取可见的移动标签
        visible_movings = [tag_id for tag_id in self.moving_tag_ids if tag_id in tag_poses]
        
        # 需要至少两个移动标签可见才能更新关系
        if len(visible_movings) < 2:
            return False
            
        # 选择第一个移动标签作为基准
        base_tag_id = visible_movings[0]
        base_pos, base_rot = tag_poses[base_tag_id]
        
        # 更新相对位姿关系
        for tag_id in visible_movings[1:]:
            curr_pos, curr_rot = tag_poses[tag_id]
            
            # 计算相对位置
            rel_pos = curr_pos - base_pos
            
            # 计算相对旋转
            rel_rot = np.dot(curr_rot, base_rot.T)
            
            # 存储或更新相对位姿
            self.moving_tag_relations[tag_id] = (rel_pos, rel_rot)
            
        return True 