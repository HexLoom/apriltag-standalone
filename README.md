# AprilTag独立检测库

这是一个基于pupil-apriltags库的AprilTag识别工具包，用于摄像头中AprilTag的检测和跟踪。

## 依赖库

- Python 3.6+
- OpenCV
- NumPy
- pupil-apriltags

## 安装

1. 确保已安装Python环境
2. 安装必要的依赖：

```bash
pip install opencv-python numpy pupil-apriltags
```

## 使用方法

### 基本用法

```python
import cv2
from apriltag import Detector, DetectorOptions

# 创建检测器
options = DetectorOptions(
    families="tag36h11",  # 标签家族
    border=1,             # 标签边框大小
    nthreads=4,           # 线程数量
    quad_decimate=1.0,    # 图像下采样系数
    quad_blur=0.0,        # 高斯模糊系数
    refine_edges=True     # 是否精细化边缘
)
detector = Detector(options)

# 读取图像
img = cv2.imread("test_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测AprilTag
detections = detector.detect(gray)

# 显示检测结果
for detection in detections:
    print(f"标签家族: {detection.tag_family}, ID: {detection.tag_id}")
    print(f"位置: {detection.center}")
    print(f"角点: {detection.corners}")
```

### 绘制检测结果

```python
import numpy as np
from apriltag import draw_detection_results

# 相机内参矩阵和畸变系数
K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
D = np.zeros((4, 1), dtype=np.float32)

# 绘制检测结果
result_img = draw_detection_results(img, detections, K, D, tag_size=0.1)

# 显示结果
cv2.imshow("AprilTag检测", result_img)
cv2.waitKey(0)
```

### 运行测试脚本

提供了一个简单的测试脚本，可以用于验证AprilTag检测功能：

```bash
python test_apriltag.py
```

这将打开电脑默认摄像头并实时检测AprilTag。按"q"键退出。

## 支持的标签家族

pupil-apriltags库支持以下标签家族：
- tag36h11 (默认)
- tag25h9
- tag16h5
- tagCircle21h7
- tagCircle49h12
- tagStandard41h12
- tagStandard52h13
- tagCustom48h12

## 注意事项

- 为获得更好的性能，可以调整DetectorOptions中的参数
- 对于计算资源有限的设备，可以考虑增加quad_decimate参数来降低计算复杂度
- 确保使用的AprilTag标记的尺寸与代码中的tag_size参数匹配
- 绘制3D坐标轴需要准确的相机参数

## 功能特点

- 支持USB摄像头实时AprilTag检测
- 计算并显示标签的3D位姿(位置和方向)
- 支持保存原始和标记后的图像
- 可自定义配置和相机参数
- 包含完整的相机标定工具
- 不依赖ROS，是原ROS包的纯Python独立版本

## 安装步骤

### 1. 安装AprilTag C库

AprilTag的C库是必需的。请按照以下步骤安装：

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y libapriltag-dev
```

#### Windows:
Windows用户需要自行编译或下载预编译的二进制文件，并确保`apriltag.dll`在系统PATH中或当前目录。

### 2. 安装Python依赖

```bash
pip install -r requirements.txt  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install pupil-apriltags -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 使用说明

### 快速开始 (推荐)

最简单的使用方式是运行集成工具，它提供了交互式菜单来引导您完成所有步骤：

```bash
python apriltag_tool.py
```

这个工具会提供菜单选项：
1. 生成棋盘格标定板
2. 相机标定
3. AprilTag检测
4. 查看帮助文档

只需按照菜单提示操作即可完成整个流程。

### 相机标定

在使用AprilTag检测前，建议先进行相机标定，获取准确的相机参数：

```bash
# 首先生成棋盘格标定板
python create_chessboard.py --size 9x6 --square 100 --output chessboard.png --dpi 300

# 打印棋盘格并测量实际方格大小，然后进行标定
python camera_calibration.py --size 9x6 --square 0.025 --output config/camera/camera_info_1.yaml
```

参数说明:

**棋盘格生成工具 (create_chessboard.py):**
- `--size`: 棋盘格内角点数量，宽x高 (默认: 9x6)
- `--square`: 方格大小，像素 (默认: 100)
- `--output`: 输出文件路径 (默认: chessboard.png)
- `--dpi`: 输出图像的DPI (默认: 300)，影响打印尺寸

**相机标定程序 (camera_calibration.py):**
- `--size`: 棋盘格内角点数量，宽x高 (默认: 9x6)
- `--square`: 棋盘格方块大小，单位米 (默认: 0.025)
- `--output`: 输出文件路径 (默认: config/camera/camera_info_1.yaml)
- `--camera`: 摄像头设备ID (默认: 0)
- `--width`: 摄像头捕获宽度 (默认: 1280)
- `--height`: 摄像头捕获高度 (默认: 720)
- `--samples`: 标定所需样本数量 (默认: 20)
- `--preview`: 标定完成后预览校正效果

标定过程：
1. 生成并打印棋盘格标定板
2. 测量实际方格大小（以米为单位）
3. 运行标定程序，将棋盘格放在相机前，从不同角度采集样本
4. 程序会自动检测棋盘格并收集样本，也可按's'键手动保存当前帧
5. 收集足够样本后，程序自动计算相机参数并保存到指定文件

### AprilTag检测

标定完成后，可以运行AprilTag检测程序：

```bash
python apriltag_detector.py
```

### 高级用法

```bash
python apriltag_detector.py [配置文件路径] --camera 相机ID --width 宽度 --height 高度 --camera_info 相机参数文件
```

参数说明:
- `配置文件路径`: AprilTag配置文件路径 (默认: `config/vision/tags_36h11_all.json`)
- `--camera`: 摄像头设备ID (默认: 0)
- `--camera_info`: 相机内参文件路径 (默认: `config/camera/camera_info_1.yaml`)
- `--width`: 摄像头捕获宽度 (默认: 1280)
- `--height`: 摄像头捕获高度 (默认: 720)

### 按键控制

- `q`: 退出程序

## 配置文件说明

### AprilTag配置文件 (JSON)

位于 `config/vision/tags_36h11_all.json`，包含以下主要配置:

```json
{
    "AprilTagConfig": {
        "family": "tag36h11",      // 标签家族
        "size": 0.1,               // 标签尺寸(米)
        "threads": 2,              // 线程数
        // 其他检测参数...
    },

    "Archive": {
        "enable": true,            // 是否启用存档
        "preview": true,           // 是否实时预览
        "path": "./data/apriltag"  // 存档路径
        // 其他存档参数...
    }
}
```
✅ AprilTagConfig 部分：这是关于 AprilTag 检测器本身的配置。

参数	含义
family: "tag36h11"	使用的 AprilTag 标签家族，tag36h11 是一种常见的、高精度标签家族。
size: 0.1	标签在现实世界中的实际边长（单位：米），通常用于从图像中恢复真实的三维位置。
threads: 2	用于并行处理的线程数量，加快检测速度。
max_hamming: 0	最大汉明距离，表示识别时容忍的标签编码误差数，0 表示只接受完全匹配的标签。
z_up: true	坐标系的方向设置，若为 true，表示 Z 轴朝上（适用于某些 SLAM 系统）。
图像预处理参数：

参数	含义
decimate: 1.0	图像降采样系数，1.0 表示不降采样；设置小于1可以加速但降低精度。
blur: 1.0	应用高斯模糊的程度，用于减少噪声。
refine_edges: 1	是否优化检测边缘，1 表示开启，提升精度但增加计算量。
debug: 0	是否开启调试信息，0 表示关闭调试输出。
📁 Archive 部分：控制检测结果的保存与预览。

参数	含义
enable: true	是否开启归档功能（保存图像和检测结果）。
preview: true	检测后是否显示检测结果的图像预览窗口。
preview_delay: 3000	预览图像显示的时间（毫秒），这里是 3 秒。
save_raw: false	是否保存原始输入图像，false 表示不保存。
save_pred: true	是否保存包含检测结果的图像（带标签边框和ID）。
path: "./data/apriltag"	保存图像和数据的路径。

### 相机参数文件 (YAML)

位于 `config/camera/camera_info_1.yaml`，包含相机内参和畸变系数：

```yaml
image_width: 1280
image_height: 720
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distortion_coefficients:
  rows: 1
  cols: 5
  data: [k1, k2, p1, p2, k3]
# 其他参数...
```

## 常见问题

1. **找不到apriltag库**
   
   确保已正确安装apriltag库，并且库文件在系统中可以找到。

2. **相机无法打开**
   
   检查相机设备ID是否正确，以及相机是否被其他程序占用。

3. **检测结果不准确**
   
   确保您的相机已正确标定，并且配置文件中的标签尺寸正确。

4. **标定过程中找不到棋盘格**
   
   确保棋盘格干净平整，光照均匀，避免强反光。尽量使棋盘格占据画面较大部分。

## 文件结构说明

```
apriltag_standalone/
├── apriltag.py              # AprilTag检测库核心代码
├── apriltag_detector.py     # AprilTag检测主程序
├── apriltag_tool.py         # 集成工具启动菜单
├── camera_calibration.py    # 相机标定程序
├── create_chessboard.py     # 棋盘格生成工具
├── configs.py               # 配置文件处理
├── config/                  # 配置目录
│   ├── camera/              # 相机配置
│   │   └── camera_info_1.yaml  # 相机参数
│   └── vision/              # 视觉配置
│       └── tags_36h11_all.json # AprilTag配置
├── README.md                # 说明文档
└── requirements.txt         # Python依赖
```

## 技术说明

本项目是从ROS AprilTag检测包移植的独立版本，移除了ROS依赖，保留了核心功能。
主要使用了以下技术：

- OpenCV: 图像处理、相机标定和姿态估计
- AprilTag C库: 标签检测
- SciPy: 旋转矩阵和四元数转换

## 许可证

本项目基于MIT许可证 