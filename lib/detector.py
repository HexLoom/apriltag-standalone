#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AprilTag Python包装器 (使用pupil-apriltags库)

这个模块创建了两个用于检测AprilTag并提取信息的类。
通过这个模块，你可以识别图像中所有可见的AprilTag，
并获取有关标签位置和方向的信息。

原作者: Isaac Dulin, Spring 2016
更新: Matt Zucker, Fall 2016
精简和修改: 用于独立版本, 2023
修改为使用pupil-apriltags库, 2023
"""

import collections
import numpy as np
import cv2
import pupil_apriltags as apriltags

# 定义检测结果类
DetectionBase = collections.namedtuple(
    'DetectionBase',
    'tag_family, tag_id, hamming, goodness, decision_margin, '
    'homography, center, corners')

class Detection(DetectionBase):
    '''AprilTag检测结果类，继承自命名元组'''

    _print_fields = [
        'Family', 'ID', 'Hamming error', 'Goodness',
        'Decision margin', 'Homography', 'Center', 'Corners'
    ]

    _max_len = max(len(field) for field in _print_fields)

    def tostring(self, values=None, indent=0):
        '''将对象转为字符串'''
        rval = []
        indent_str = ' '*(self._max_len+2+indent)

        if not values:
            values = collections.OrderedDict(zip(self._print_fields, self))

        for label in values:
            value_str = str(values[label])

            if value_str.find('\n') > 0:
                value_str = value_str.split('\n')
                value_str = [value_str[0]] + [indent_str+v for v in value_str[1:]]
                value_str = '\n'.join(value_str)

            rval.append('{:>{}s}: {}'.format(
                label, self._max_len+indent, value_str))

        return '\n'.join(rval)

    def __str__(self):
        '''字符串表示'''
        return self.tostring()

# 检测器选项类
class DetectorOptions(object):
    '''
    检测器配置选项类
    '''
    def __init__(self,
                 families='tag36h11',
                 border=1,
                 nthreads=4,
                 quad_decimate=1.0,
                 quad_blur=0.0,
                 refine_edges=True,
                 refine_decode=False,
                 refine_pose=False,
                 debug=False,
                 quad_contours=True):
        self.families = families
        self.border = int(border)
        self.nthreads = int(nthreads)
        self.quad_decimate = float(quad_decimate)
        self.quad_sigma = float(quad_blur)
        self.refine_edges = int(refine_edges)
        self.refine_decode = int(refine_decode)
        self.refine_pose = int(refine_pose)
        self.debug = int(debug)
        self.quad_contours = quad_contours

# 检测器类
class Detector(object):
    '''AprilTag检测器类'''
    
    def __init__(self, options=None, searchpath=[]):
        '''
        初始化检测器
        
        参数:
            options: 检测器选项
            searchpath: 在使用pupil-apriltags时不需要，保留参数以兼容旧代码
        '''
        if options is None:
            options = DetectorOptions()
            
        self.options = options
        
        # 初始化标签家族列表
        self.tag_families = []
        
        # 可用的tag家族
        if isinstance(options.families, str):
            # 如果是字符串，拆分并保存到列表
            self.tag_families = options.families.split(',')
            families_str = options.families  # 保持字符串格式
        else:
            # 如果已经是列表，直接使用
            self.tag_families = options.families
            families_str = ','.join(self.tag_families)  # 转为字符串格式
        
        # 创建pupil-apriltags检测器
        self.detector = apriltags.Detector(
            families=families_str,  # 使用字符串格式
            nthreads=options.nthreads,
            quad_decimate=options.quad_decimate,
            quad_sigma=options.quad_sigma,
            refine_edges=options.refine_edges,
            decode_sharpening=options.refine_decode,
            debug=options.debug
        )
        
        print(f"使用pupil-apriltags库创建了检测器，支持的标签家族: {self.tag_families}")
    
    def detect(self, img, return_image=False):
        '''
        在图像中检测AprilTag
        
        参数:
            img: 灰度图像(numpy数组)
            return_image: 是否返回标记了检测结果的图像
            
        返回:
            检测结果列表
        '''
        # 确保图像是灰度图
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用pupil-apriltags检测
        pupil_detections = self.detector.detect(img)
        
        # 创建与原格式兼容的检测结果
        apriltag_detections = []
        
        for det in pupil_detections:
            # 从pupil-apriltags检测结果创建与原格式兼容的Detection对象
            apriltag_detections.append(Detection(
                tag_family=det.tag_family,
                tag_id=det.tag_id,
                hamming=det.hamming,
                goodness=0.0,  # pupil-apriltags不提供此值
                decision_margin=det.decision_margin,
                homography=det.homography,
                center=det.center,
                corners=det.corners))
        
        # 如果需要返回带标记的图像
        if return_image:
            # 将检测结果标记在图像上
            annotated_img = self._vis_detections(img.shape, apriltag_detections)
            return apriltag_detections, annotated_img
        else:
            return apriltag_detections
    
    def add_tag_family(self, name):
        '''添加标签家族（兼容接口，实际创建时已指定）'''
        print(f"注意: 使用pupil-apriltags时，标签家族应在创建检测器时指定。")
    
    def _vis_detections(self, shape, detections):
        '''将检测结果可视化'''
        # 创建一个RGB图像
        img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        
        # 绘制每个检测
        for det in detections:
            # 绘制边框
            for i in range(4):
                j = (i + 1) % 4
                p1 = (int(det.corners[i][0]), int(det.corners[i][1]))
                p2 = (int(det.corners[j][0]), int(det.corners[j][1]))
                cv2.line(img, p1, p2, (0, 255, 0), 2)
            
            # 绘制中心点
            center = (int(det.center[0]), int(det.center[1]))
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            
            # 添加标签
            text_pos = (int(det.corners[0][0]), int(det.corners[0][1] - 10))
            cv2.putText(img, f"{det.tag_family} {det.tag_id}", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img


def draw_detection_results(image, detections, K, D, tag_size=0.1, flip_z=False):
    """
    在图像上绘制AprilTag检测结果
    
    参数:
        image: 输入图像
        detections: 检测结果列表
        K: 相机内参矩阵
        D: 相机畸变系数
        tag_size: AprilTag实际尺寸(米)
        flip_z: 是否翻转Z轴方向
        
    返回:
        带标记的图像
    """
    
    # 复制图像，以免修改原图
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    for tag in detections:
        # 提取角点
        (ptA, ptB, ptC, ptD) = tag.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        
        # 相机坐标系中的角点
        camera_corners = np.array(tag.corners, dtype=np.float32)
        
        # 世界坐标系中的角点
        world_corners = np.array([
            [-tag_size/2, tag_size/2, 0],
            [tag_size/2, tag_size/2, 0],
            [tag_size/2, -tag_size/2, 0],
            [-tag_size/2, -tag_size/2, 0]
        ], dtype=np.float32)
        
        # 计算位姿 - 使用SOLVEPNP_IPPE_SQUARE算法，更适合方形标记
        _, rvec, tvec = cv2.solvePnP(
            world_corners, 
            camera_corners, 
            K, D, 
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        # 绘制坐标轴 - 确保绘制的坐标轴长度与标签尺寸匹配
        cv2.drawFrameAxes(output, K, D, rvec, tvec, tag_size/2)
        
        # 绘制检测框
        cv2.line(output, ptA, ptB, (0, 255, 0), 2)
        cv2.line(output, ptB, ptC, (0, 255, 0), 2)
        cv2.line(output, ptC, ptD, (0, 255, 0), 2)
        cv2.line(output, ptD, ptA, (0, 255, 0), 2)
        
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 计算欧拉角 (以度为单位)
        from scipy.spatial.transform import Rotation as R_scipy
        r = R_scipy.from_matrix(R)
        euler = r.as_euler('xyz', degrees=True)
        
        # 计算中心位置用于文本放置
        center_x = int(np.mean([ptA[0], ptB[0], ptC[0], ptD[0]]))
        center_y = int(np.mean([ptA[1], ptB[1], ptC[1], ptD[1]]))
        
        # 在标签上方显示ID和位姿信息
        cv2.putText(output, f"ID: {tag.tag_id}", 
                    (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
                    
        # 位置信息 (取前三位小数)
        text_color = (0, 255, 0)
        pos_x = f"X: {tvec[0,0]:.3f}m"
        pos_y = f"Y: {tvec[1,0]:.3f}m"
        pos_z = f"Z: {tvec[2,0]:.3f}m"
        
        # 欧拉角信息
        angle_x = f"Roll: {euler[0]:.1f}"
        angle_y = f"Pitch: {euler[1]:.1f}"
        angle_z = f"Yaw: {euler[2]:.1f}"
        
        # 根据图像宽度调整文本位置
        text_offset = 15
        if center_x < output.shape[1]//2:  # 左侧
            text_x = ptB[0] + 10
        else:  # 右侧
            text_x = ptA[0] - 120
            
        cv2.putText(output, pos_x, 
                    (text_x, ptA[1] + text_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
        cv2.putText(output, pos_y, 
                    (text_x, ptA[1] + 2*text_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
        cv2.putText(output, pos_z, 
                    (text_x, ptA[1] + 3*text_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
        
        # 添加角度信息
        angle_color = (255, 0, 0)  # 蓝色 (Blue, Green, Red)
        if center_y < output.shape[0]//2:  # 上半部分
            angle_text_y = ptD[1] + text_offset
        else:  # 下半部分
            angle_text_y = ptA[1] - 4*text_offset
            
        cv2.putText(output, angle_x, 
                    (text_x, angle_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, angle_color, 1)
        cv2.putText(output, angle_y, 
                    (text_x, angle_text_y + text_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, angle_color, 1)
        cv2.putText(output, angle_z, 
                    (text_x, angle_text_y + 2*text_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, angle_color, 1)
        
    # 在右下角添加检测数量信息
    detection_info = f"AprilTag:{len(detections)} "
    cv2.putText(output, detection_info, 
                (output.shape[1] - 100, output.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    return output 