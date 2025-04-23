#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AprilTag工具集合

整合所有功能的启动菜单，方便用户使用

使用方法:
    python apriltag_tool.py
"""

import os
import sys
import subprocess
import argparse

from lib.filesystem import create_dirs_if_not_exist

def clear_screen():
    """清空控制台屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """打印标题"""
    clear_screen()
    print("=" * 70)
    print("                        AprilTag 工具集合")
    print("=" * 70)
    print("本工具整合了AprilTag检测和相机标定的全部功能\n")

def print_menu():
    """打印主菜单"""
    print("请选择功能:")
    print("  1. 生成棋盘格标定板")
    print("  2. 相机标定")
    print("  3. AprilTag检测")
    print("  4. 查看帮助文档")
    print("  0. 退出")
    print("\n输入选项编号: ", end="")

def run_script(script_name, args=None):
    """运行Python脚本"""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n脚本已中断")
    except Exception as e:
        print(f"\n运行错误: {e}")
    
    input("\n按回车键返回主菜单...")

def chessboard_generation():
    """棋盘格生成向导"""
    clear_screen()
    print("=" * 70)
    print("                      棋盘格标定板生成向导")
    print("=" * 70)
    print("生成用于相机标定的棋盘格图案\n")
    
    try:
        size = input("输入棋盘格内角点数量 (宽x高, 默认 9x6): ") or "9x6"
        square = input("输入方格大小 (像素, 默认 100): ") or "100"
        dpi = input("输入DPI (默认 300): ") or "300"
        output = input("输入输出文件路径 (默认 chessboard.png): ") or "chessboard.png"
        
        # 转换为参数列表
        args = [
            "--size", size,
            "--square", square,
            "--dpi", dpi, 
            "--output", output
        ]
        
        run_script("lib/create_chessboard.py", args)
    except KeyboardInterrupt:
        print("\n操作已取消")
        input("\n按回车键返回主菜单...")

def camera_calibration():
    """相机标定向导"""
    clear_screen()
    print("=" * 70)
    print("                         相机标定向导")
    print("=" * 70)
    print("使用棋盘格标定板进行相机标定\n")
    
    try:
        # 确保配置目录存在
        create_dirs_if_not_exist("config/camera")
        
        size = input("输入棋盘格内角点数量 (宽x高, 默认 9x6): ") or "9x6"
        square = input("输入实际方格大小 (米, 默认 0.025): ") or "0.025"
        camera = input("输入相机设备ID (默认 0): ") or "0"
        width = input("输入相机宽度 (默认 1920): ") or "1920"
        height = input("输入相机高度 (默认 1080): ") or "1080"
        samples = input("输入标定样本数量 (默认 20): ") or "20"
        output = input("输入输出文件路径 (默认 config/camera/camera_info_1.yaml): ") or "config/camera/camera_info_1.yaml"
        preview = input("标定后是否预览效果 (y/n, 默认 y): ").lower() in ("y", "yes", "") 
        
        # 转换为参数列表
        args = [
            "--size", size,
            "--square", square,
            "--camera", camera, 
            "--width", width,
            "--height", height,
            "--samples", samples,
            "--output", output
        ]
        
        if preview:
            args.append("--preview")
        
        run_script("lib/camera_calibration.py", args)
    except KeyboardInterrupt:
        print("\n操作已取消")
        input("\n按回车键返回主菜单...")

def apriltag_detection():
    """AprilTag检测向导"""
    clear_screen()
    print("=" * 70)
    print("                      AprilTag 检测向导")
    print("=" * 70)
    print("使用USB相机检测AprilTag并计算位姿\n")
    
    try:
        # 确保配置目录存在
        create_dirs_if_not_exist("config/vision")
        
        # 检查相机标定文件是否存在
        default_camera_info = "config/camera/camera_info_1.yaml"
        if not os.path.exists(default_camera_info):
            print(f"警告: 默认相机标定文件 {default_camera_info} 不存在!")
            print("建议先进行相机标定，否则检测结果可能不准确。")
            proceed = input("是否继续? (y/n, 默认 n): ").lower() in ("y", "yes")
            if not proceed:
                return
        
        # 检查标签配置文件是否存在
        default_tag_config = "config/vision/tags_36h11_all.json"
        if not os.path.exists(default_tag_config):
            print(f"警告: 默认标签配置文件 {default_tag_config} 不存在!")
            print("将使用配置文件中的默认值。")
        
        camera = input("输入相机设备ID (默认 0): ") or "0"
        width = input("输入相机宽度 (默认 1920): ") or "1920"
        height = input("输入相机高度 (默认 1080): ") or "1080"
        config_path = input("输入标签配置文件路径 (默认 config/vision/tags_36h11_all.json): ") or "config/vision/tags_36h11_all.json"
        camera_info = input("输入相机标定文件路径 (默认 config/camera/camera_info_1.yaml): ") or "config/camera/camera_info_1.yaml"
        
        # 转换为参数列表
        args = [
            config_path,
            "--camera", camera, 
            "--width", width,
            "--height", height,
            "--camera_info", camera_info
        ]
        
        run_script("lib/apriltag_detector.py", args)
    except KeyboardInterrupt:
        print("\n操作已取消")
        input("\n按回车键返回主菜单...")

def show_help():
    """显示帮助信息"""
    clear_screen()
    print("=" * 70)
    print("                         帮助文档")
    print("=" * 70)
    print("AprilTag工具集使用说明:\n")
    
    print("1. 标定流程:")
    print("   a. 首先生成并打印棋盘格标定板")
    print("   b. 测量实际方格大小（以米为单位）")
    print("   c. 使用相机标定工具进行相机标定")
    print("   d. 标定过程中从不同角度采集样本，确保覆盖相机视野")
    print("   e. 标定完成后，标定结果会保存为YAML文件")
    
    print("\n2. AprilTag检测:")
    print("   a. 确保已完成相机标定并有标定文件")
    print("   b. 运行AprilTag检测工具")
    print("   c. 将AprilTag标签放在相机视野内")
    print("   d. 程序会实时显示检测结果和位姿信息")
    
    print("\n3. 配置文件:")
    print("   - 相机标定文件: config/camera/camera_info_1.yaml")
    print("   - AprilTag配置: config/vision/tags_36h11_all.json")
    
    print("\n4. 注意事项:")
    print("   - 标定时，棋盘格应保持平整")
    print("   - 检测时，确保标签尺寸在配置文件中设置正确")
    print("   - 相机配置准确对位姿估计精度至关重要")
    print("   - 在低光或强光下性能可能会降低")
    
    input("\n按回车键返回主菜单...")

def main():
    """主函数"""
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input().strip()
            
            if choice == "1":
                chessboard_generation()
            elif choice == "2":
                camera_calibration()
            elif choice == "3":
                apriltag_detection()
            elif choice == "4":
                show_help()
            elif choice == "0":
                print("\n感谢使用AprilTag工具集，再见!")
                break
            else:
                print("\n无效选项，请重新输入!")
                input("\n按回车键继续...")
        except KeyboardInterrupt:
            print("\n\n程序已中断，再见!")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            input("\n按回车键继续...")

if __name__ == "__main__":
    main() 