# slm_image_processing.py
import numpy as np
import cv2
import os

# 确保保存目录存在
save_dir = "SLM_phase_images"
os.makedirs(save_dir, exist_ok=True)

def create_slm_phase_for_convolution(kernel_name='sharpen', add_grating=True):
    """
    快速创建用于图像处理的SLM相位图
    """
    # SLM参数（FSLM-2K70-P02HR）
    W, H = 1920, 1080
    pixel_size = 8e-6

    # 生成坐标
    x = np.linspace(-W / 2, W / 2, W)
    y = np.linspace(-H / 2, H / 2, H)
    X, Y = np.meshgrid(x, y)

    # 定义卷积核
    kernels = {
        'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'edge': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'blur': np.ones((3, 3)) / 9,
        'gaussian': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        'laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    }

    kernel = kernels.get(kernel_name, kernels['sharpen'])

    # 将核扩展到SLM分辨率
    kernel_full = np.zeros((H, W))
    kh, kw = kernel.shape
    kernel_full[H // 2 - kh // 2:H // 2 + kh // 2 + 1, W // 2 - kw // 2:W // 2 + kw // 2 + 1] = kernel

    # 计算傅里叶变换
    kernel_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(kernel_full)))

    # 提取相位
    phase = np.angle(kernel_ft)

    # 添加闪耀光栅（消除零级光）
    if add_grating:
        grating_period = 50  # 像素
        grating = 2 * np.pi * (X % grating_period) / grating_period
        phase = (phase + grating) % (2 * np.pi)

    # 归一化到0-255
    phase_8bit = (phase / (2 * np.pi) * 255).astype(np.uint8)

    # 保存图像
    filename = f"slm_{kernel_name}_phase.bmp"
    save_path = os.path.join(save_dir, filename)  # 构建完整路径
    cv2.imwrite(save_path, phase_8bit)
    print(f"Generated {save_path} for {kernel_name} operation")

    return phase_8bit


# 生成各种处理的相位图
if __name__ == "__main__":
    operations = ['sharpen', 'edge', 'blur', 'gaussian', 'laplacian']

    for op in operations:
        create_slm_phase_for_convolution(op)

    print("所有相位图已生成，可通过SLM控制软件加载使用")