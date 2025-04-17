#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化模块 - 提供3D可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

class TableVisualizer:
    def __init__(self):
        # 初始化坐标范围和图形存储变量
        # 自动调整坐标范围
        self.x_min = -0.2
        self.x_max = 0.7
        self.y_min = -0.2
        self.y_max = 0.7
        self.z_min = -0.2
        self.z_max = 0.5
        
        # 是否需要更新坐标范围
        self.needs_axis_update = False
        
        # 存储图形对象
        self.quivers = []
        self.texts = []
        self.scatter_points = []
        self.table_lines = []
        
        # 标记移动轨迹
        self.track_history = []
        self.track_line = None
        
        # 创建绘图窗口和3D坐标系
        plt.ion()  # 启用交互模式
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_axes()

    def setup_axes(self):
        """初始化3D坐标系"""
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max) 
        self.ax.set_zlim(self.z_min, self.z_max)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('AprilTag Table Tracking')
        self.ax.view_init(elev=30, azim=45)
        
        # 绘制世界坐标系
        self.ax.quiver(0, 0, 0, 0.1, 0, 0, color='r', linewidth=2)  # X轴
        self.ax.quiver(0, 0, 0, 0, 0.1, 0, color='g', linewidth=2)  # Y轴
        self.ax.quiver(0, 0, 0, 0, 0, 0.1, color='b', linewidth=2)  # Z轴

    def update_axis_limits(self, tag_poses):
        """根据检测到的标签自动调整坐标范围"""
        if not tag_poses:
            return
            
        # 收集所有标签位置
        positions = [t for t, _ in tag_poses.values()]
        if not positions:
            return
            
        # 计算当前边界
        pos_array = np.array(positions)
        x_min_current = np.min(pos_array[:, 0]) - 0.3
        x_max_current = np.max(pos_array[:, 0]) + 0.3
        y_min_current = np.min(pos_array[:, 1]) - 0.3
        y_max_current = np.max(pos_array[:, 1]) + 0.3
        z_min_current = np.min(pos_array[:, 2]) - 0.2
        z_max_current = np.max(pos_array[:, 2]) + 0.3
        
        # 检查是否需要更新
        if (x_min_current < self.x_min or x_max_current > self.x_max or
            y_min_current < self.y_min or y_max_current > self.y_max or
            z_min_current < self.z_min or z_max_current > self.z_max):
            
            # 更新范围（稍微扩大一点）
            self.x_min = min(self.x_min, x_min_current)
            self.x_max = max(self.x_max, x_max_current)
            self.y_min = min(self.y_min, y_min_current)
            self.y_max = max(self.y_max, y_max_current)
            self.z_min = min(self.z_min, z_min_current)
            self.z_max = max(self.z_max, z_max_current)
            
            # 设置需要更新标志
            self.needs_axis_update = True

    def draw_table(self, reference_tags_poses):
        """绘制桌面 - 根据实际检测到的参考标签绘制，不固定顺序"""
        # 清除旧的桌面线
        while self.table_lines:
            self.table_lines.pop(0).remove()
            
        # 确保至少有3个参考标签可见
        if len(reference_tags_poses) < 3:
            return
            
        # 计算参考标签的中心点坐标
        tag_positions = [pos for pos, _ in reference_tags_poses.values()]
        
        # 尝试通过凸包算法找到桌子边缘
        if len(tag_positions) >= 3:
            try:
                # 提取x,y坐标来形成2D平面
                points_2d = np.array([[p[0], p[1]] for p in tag_positions])
                
                # 计算2D凸包
                hull = ConvexHull(points_2d)
                
                # 绘制凸包边缘
                for simplex in hull.simplices:
                    i, j = simplex
                    line, = self.ax.plot3D(
                        [tag_positions[i][0], tag_positions[j][0]],
                        [tag_positions[i][1], tag_positions[j][1]],
                        [tag_positions[i][2], tag_positions[j][2]],
                        'gray', linewidth=1, alpha=0.5)
                    self.table_lines.append(line)
            except Exception as e:
                # 如果凸包计算失败，退回到简单连线
                print(f"构建桌面轮廓失败: {e}")
                self.draw_simple_table(tag_positions)
        else:
            self.draw_simple_table(tag_positions)
    
    def draw_simple_table(self, tag_positions):
        """简单地连接所有标签点"""
        # 按标签ID排序
        n = len(tag_positions)
        for i in range(n):
            j = (i + 1) % n
            line, = self.ax.plot3D(
                [tag_positions[i][0], tag_positions[j][0]],
                [tag_positions[i][1], tag_positions[j][1]],
                [tag_positions[i][2], tag_positions[j][2]],
                'gray', linewidth=1, alpha=0.5)
            self.table_lines.append(line)

    def update(self, tag_poses, moving_tag_id, reference_tags):
        """更新3D显示"""
        # 检查是否需要调整坐标范围
        self.update_axis_limits(tag_poses)
        if self.needs_axis_update:
            self.ax.set_xlim(self.x_min, self.x_max)
            self.ax.set_ylim(self.y_min, self.y_max) 
            self.ax.set_zlim(self.z_min, self.z_max)
            self.needs_axis_update = False
        
        # 清除旧图形
        while self.quivers:
            self.quivers.pop(0).remove()
        while self.texts:
            self.texts.pop(0).remove()
        while self.scatter_points:
            self.scatter_points.pop(0).remove()

        # 分离参考标签和移动标签
        reference_tags_poses = {id: pose for id, pose in tag_poses.items() 
                              if id in reference_tags}
        
        # 绘制桌面
        self.draw_table(reference_tags_poses)
        
        # 绘制每个标签
        for tag_id, (t, R) in tag_poses.items():
            if tag_id == moving_tag_id:  # 特殊处理移动标签
                color = 'magenta' # 移动标签颜色
                axis_scale = 0.1 # 坐标轴长度
                scatter_size = 100 # 位置点大小
                
                # 记录移动标签轨迹
                self.track_history.append(t.copy())
                if len(self.track_history) > 100:  # 限制历史长度
                    self.track_history.pop(0)
                    
                # 更新轨迹线
                if self.track_line:
                    self.track_line.remove()
                if len(self.track_history) > 1:
                    track_array = np.array(self.track_history)
                    self.track_line, = self.ax.plot3D(
                        track_array[:, 0], track_array[:, 1], track_array[:, 2],
                        'red', linewidth=1, alpha=0.5)
            else:
                color = 'cyan' # 其他标签颜色
                axis_scale = 0.05 # 坐标轴长度
                scatter_size = 70 # 位置点大小
                
            # 绘制位置点
            scatter = self.ax.scatter(t[0], t[1], t[2], c=color, s=scatter_size)
            self.scatter_points.append(scatter)

            # 绘制坐标系轴
            x_end = t + R[:,0] * axis_scale
            y_end = t + R[:,1] * axis_scale
            z_end = t + R[:,2] * axis_scale

            # X轴 (红色)
            self.quivers.append(self.ax.quiver(
                t[0], t[1], t[2],
                x_end[0]-t[0], x_end[1]-t[1], x_end[2]-t[2],
                color='red', linewidth=1.5))

            # Y轴 (绿色)
            self.quivers.append(self.ax.quiver(
                t[0], t[1], t[2],
                y_end[0]-t[0], y_end[1]-t[1], y_end[2]-t[2],
                color='green', linewidth=1.5))

            # Z轴 (蓝色)
            self.quivers.append(self.ax.quiver(
                t[0], t[1], t[2],
                z_end[0]-t[0], z_end[1]-t[1], z_end[2]-t[2],
                color='blue', linewidth=1.5))

            # 添加标签ID (使用英文)
            text = self.ax.text(t[0], t[1], t[2]+0.03, str(tag_id),
                              color='black', fontsize=10, ha='center')
            self.texts.append(text)

        # 刷新图形
        self.fig.canvas.draw()
        plt.pause(0.001) 