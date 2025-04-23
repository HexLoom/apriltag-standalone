# 桌面AprilTag跟踪系统

## 项目概述

这个程序用于跟踪桌面上的多个AprilTag标签，包括四个角落固定的参考标签(ID 0-3)以及一个可移动的标签(ID 4)。程序会计算移动标签相对于参考标签的位置，并使用3D可视化展示所有标签的空间关系。

## 模块化结构

该项目采用模块化设计，将功能分散到不同的模块中，便于维护和扩展：

- `table_tracking.py`: 主程序入口，处理命令行参数并启动跟踪
- `modules/`: 模块化代码目录
  - `__init__.py`: 包定义文件
  - `config.py`: 配置模块，负责读取和管理配置项
  - `utils.py`: 工具函数模块，提供通用辅助函数
  - `visualizer.py`: 可视化模块，负责3D可视化显示
  - `tracker.py`: 跟踪处理模块，处理AprilTag标签跟踪和位姿计算
- `lib/`: 检测器库目录
- `config/`: 配置文件目录
  - `vision/`: 视觉相关配置
  - `camera/`: 相机参数配置
- `data/`: 数据存储目录

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

基本用法：

```bash
python table_tracking.py
```

带参数的用法：

```bash
python table_tracking.py --camera 0 --config config/vision/table_setup.json --camera_info config/camera/HSK_200W53_1080P.yaml --width 1280 --height 720
```

参数说明：
- `--camera`: 相机设备ID (默认: 0)
- `--config`: 配置文件路径 (默认: config/vision/table_setup.json)
- `--camera_info`: 相机参数文件 (默认: config/camera/HSK_200W53_1080P.yaml)
- `--width`: 相机宽度分辨率 (默认: 1280)
- `--height`: 相机高度分辨率 (默认: 720)

## 配置文件

配置文件使用JSON格式，包含以下主要部分：

1. `AprilTagConfig`: AprilTag检测器配置
   - `family`: 标签系列，如"tag36h11"
   - `size`: 标签物理尺寸（米）
   - `threads`: 使用的线程数
   - `z_up`: 是否Z轴向上
   - 其他检测参数

2. `TableConfig`: 桌面配置
   - `reference_tags`: 参考标签ID列表
   - `moving_tag`: 移动标签ID
   - `tag_positions`: 参考标签预设位置

3. `Archive`: 存档配置
   - `enable`: 是否启用存档
   - `preview`: 是否显示预览窗口
   - `save_raw`: 是否保存原始图像
   - `save_pred`: 是否保存处理后的图像
   - `path`: 存档路径

## 扩展和自定义

### 添加新功能

要添加新功能，可以在模块目录下创建新的模块文件，然后在主程序中导入使用。

### 修改可视化

可视化相关的设置和方法可以在`modules/visualizer.py`文件中调整。

### 自定义标签配置

可以在配置文件中修改标签ID和预设位置，以适应不同的桌面环境。

## 运行时控制

- 按 'q' 键退出程序