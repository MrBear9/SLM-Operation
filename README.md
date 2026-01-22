

# SLM Operation - 空间光调制器图像处理项目

## 项目介绍

SLMOperation 是一个基于空间光调制器（Spatial Light Modulator, SLM）的光学图像处理项目。该项目利用 PyTorch 深度学习框架，实现了对图像的各种光学卷积运算，包括图像锐化、边缘检测、模糊等操作。项目结合了光学传播原理和神经网络技术，可以将传统的数字图像处理转化为光学处理方式。

## 功能特点

### 1. 光学传播模拟
- **OpticalPropagator 类**：模拟光波在空间中的传播过程，支持 Fresnel 衍射传播
- 支持自定义波长、像素尺寸和传播距离参数
- 可以将 SLM 相位图转换为光学场分布

### 2. SLM 相位生成
- **SLM_ImageProcessor 类**：生成各种图像处理所需的 SLM 相位图
- 支持多种卷积核：sharpen、edge detection、blur、laplacian、gaussian 等
- 提供闪耀光栅（blazed grating）和 Fresnel 透镜相位生成功能
- 可视化和保存相位图功能

### 3. 神经网络相位优化
- **SLM_PhaseNet 类**：基于神经网络的 SLM 相位优化网络
- 支持输入输出通道配置
- 可用于训练优化的相位模式

### 4. 数据集处理
- **ImageProcessingDataset 类**：用于图像处理的数据集加载器
- 支持图像增强和多种操作类型

## 文件结构

```
SLMOperation/
├── OpticalPropagator.py          # 光学传播和神经网络相位模型
├── SLM_ImageProcessor.py         # SLM 图像处理器主类
├── create_slm_phase_for_convolution.py  # 创建 SLM 卷积相位
├── main.py                       # 主程序入口
├── images/                       # 测试图像目录
│   ├── bus.jpg
│   └── zidane.jpg
├── SLM_imageProcessor/           # 生成的相位图示例
│   ├── blazed_grating.bmp
│   ├── combined_phase.bmp
│   ├── edge_detection_phase.bmp
│   └── sharpen_phase.bmp
├── SLM_phase_images/             # 各种卷积核的相位图
│   ├── slm_blur_phase.bmp
│   ├── slm_edge_phase.bmp
│   ├── slm_gaussian_phase.bmp
│   ├── slm_laplacian_phase.bmp
│   └── slm_sharpen_phase.bmp
└── LICENSE                       # MIT 许可证
```

## 安装说明

### 环境要求
- Python 3.7+
- PyTorch 1.7+
- NumPy
- OpenCV (cv2)
- Matplotlib
- Pillow (PIL)

### 安装步骤

1. 克隆项目到本地：
```bash
git clone https://gitee.com/mrbear_x/slmoperation.git
cd slmoperation
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install torch numpy opencv-python matplotlib pillow
```

## 使用方法

### 1. 基本使用示例

```python
from SLM_ImageProcessor import SLM_ImageProcessor
import cv2

# 初始化 SLM 处理器
slm = SLM_ImageProcessor(
    slm_resolution=(1920, 1080),
    wavelength=532e-9,
    pixel_size=8e-6
)

# 生成锐化相位图
sharpen_phase = slm.generate_phase_for_image('sharpen')

# 可视化相位图
slm.visualize_phase_map(sharpen_phase, "Sharpening Phase")

# 保存相位图
slm.save_phase_as_image(sharpen_phase, 'sharpen_phase.bmp')
```

### 2. 创建带闪耀光栅的卷积相位

```python
from create_slm_phase_for_convolution import create_slm_phase_for_convolution

# 创建带闪耀光栅的锐化卷积相位
combined_phase = create_slm_phase_for_convolution(
    kernel_name='sharpen',
    add_grating=True
)
```

### 3. 光学传播模拟

```python
from OpticalPropagator import OpticalPropagator
import torch

# 初始化光学传播器
propagator = OpticalPropagator(
    wavelength=532e-9,
    pixel_size=8e-6,
    distance=0.1
)

# 模拟光学传播
output_field = propagator.forward(phase_slm)
```

### 4. 手动卷积操作

```python
from main import manual_convolution
import cv2

# 加载图像
image = cv2.imread('images/bus.jpg', cv2.IMREAD_GRAYSCALE)

# 定义卷积核
kernel = [[0, -1, 0],
          [-1, 5, -1],
          [0, -1, 0]]

# 执行卷积
result = manual_convolution(image, kernel)
```

### 5. 训练神经网络相位优化器

```python
from OpticalPropagator import train_neural_phase_optimizer

# 训练神经网络
train_neural_phase_optimizer()
```

## 主要类说明

### SLM_ImageProcessor

SLM 图像处理器是项目的核心类，提供以下方法：

| 方法名 | 功能描述 |
|--------|----------|
| `generate_kernel_phase(kernel_type)` | 生成指定类型的卷积核相位 |
| `generate_phase_for_image(image, operation)` | 生成图像操作的相位图 |
| `generate_blazed_grating(period, direction)` | 生成闪耀光栅相位 |
| `generate_fresnel_lens(focal_length)` | 生成 Fresnel 透镜相位 |
| `visualize_phase_map(phase_map, title)` | 可视化相位图 |
| `save_phase_as_image(phase_map, filename)` | 保存相位图为 BMP 图像 |

支持的卷积核类型：
- `sharpen`：图像锐化
- `edge`：边缘检测
- `blur`：模糊处理
- `laplacian`：拉普拉斯算子
- `gaussian`：高斯模糊

### OpticalPropagator

光学传播器类用于模拟光波传播：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| wavelength | float | 532e-9 | 激光波长（米） |
| pixel_size | float | 8e-6 | SLM 像素尺寸（米） |
| distance | float | 0.1 | 传播距离（米） |

## 许可证

本项目采用 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。

## 作者

项目作者：mrbear_x

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。