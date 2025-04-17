#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
棋盘格标定板生成工具

生成用于相机标定的棋盘格图案，可自定义尺寸和大小

使用方法:
    python create_chessboard.py [--size WIDTH HEIGHT] [--square SQUARE_SIZE] [--output OUTPUT_FILE]

参数:
    --size: 棋盘格内角点数量，宽x高 (默认: 9x6)
    --square: 方格大小，像素 (默认: 100)
    --output: 输出文件路径 (默认: chessboard.png)
    --dpi: 输出图像的DPI (默认: 300)，影响打印尺寸
"""

import numpy as np
import cv2
import argparse
import os

def create_chessboard(width, height, square_size, output_file, dpi=300):
    """
    生成棋盘格图案
    
    参数:
        width: 棋盘格宽度（内角点数量）
        height: 棋盘格高度（内角点数量）
        square_size: 方格大小（像素）
        output_file: 输出文件路径
        dpi: 输出图像的DPI
    """
    # 实际的棋盘格尺寸需要比内角点多1
    board_width = width + 1
    board_height = height + 1
    
    # 创建白色背景
    img_width = board_width * square_size
    img_height = board_height * square_size
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255
    
    # 绘制棋盘格
    for i in range(board_height):
        for j in range(board_width):
            if (i + j) % 2 == 1:  # 交替绘制黑白方格
                y1 = i * square_size
                y2 = (i + 1) * square_size
                x1 = j * square_size
                x2 = (j + 1) * square_size
                img[y1:y2, x1:x2] = 0
    
    # 创建目录（如果不存在）
    dir_path = os.path.dirname(output_file)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # 保存图像
    cv2.imwrite(output_file, img)
    
    # 计算实际打印尺寸（英寸）
    width_inch = img_width / dpi
    height_inch = img_height / dpi
    
    # 转换为厘米
    width_cm = width_inch * 2.54
    height_cm = height_inch * 2.54
    
    # 计算每个方格的实际大小（厘米）
    square_cm = square_size / dpi * 2.54
    
    print(f"棋盘格图案已生成: {output_file}")
    print(f"图案尺寸: {width}x{height} 内角点")
    print(f"图像分辨率: {img_width}x{img_height} 像素")
    print(f"DPI: {dpi}")
    print(f"打印尺寸约: {width_cm:.2f} x {height_cm:.2f} 厘米")
    print(f"每个方格约: {square_cm:.2f} 厘米")
    print(f"打印说明:")
    print(f"  1. 打印时关闭自动缩放，选择'实际大小'或'100%缩放'")
    print(f"  2. 打印后用尺子测量实际方格大小，在进行标定时作为参数输入")
    print(f"  3. 建议将图案粘贴在硬纸板上，保持平整")

def main():
    parser = argparse.ArgumentParser(description='棋盘格标定板生成工具')
    parser.add_argument('--size', type=str, default='9x6',
                        help='棋盘格内角点数量，宽x高 (默认: 9x6)')
    parser.add_argument('--square', type=int, default=100,
                        help='方格大小，像素 (默认: 100)')
    parser.add_argument('--output', type=str, default='chessboard.png',
                        help='输出文件路径 (默认: chessboard.png)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='输出图像的DPI (默认: 300)')
    args = parser.parse_args()
    
    # 解析棋盘格尺寸
    try:
        width, height = map(int, args.size.split('x'))
    except ValueError:
        print(f"错误: 棋盘格尺寸格式不正确。应为'宽x高'，例如'9x6'")
        return
    
    create_chessboard(width, height, args.square, args.output, args.dpi)

if __name__ == '__main__':
    main() 