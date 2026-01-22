import numpy as np
import cv2,os
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift


class SLM_ImageProcessor:
    def __init__(self, slm_resolution=(1920, 1080), wavelength=532e-9, pixel_size=8e-6):
        """
        初始化SLM图像处理器
        型号：FSLM-2K70-P02HR
        参数：
            - 分辨率：1920×1080
            - 像元大小：8.0μm
            - 相位范围：2π @ 633nm
        """
        self.slm_width, self.slm_height = slm_resolution
        self.wavelength = wavelength
        self.pixel_size = pixel_size

        # 创建坐标网格
        self.x = np.linspace(-self.slm_width / 2, self.slm_width / 2, self.slm_width)
        self.y = np.linspace(-self.slm_height / 2, self.slm_height / 2, self.slm_height)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 频率坐标（用于傅里叶平面）
        self.fx = np.fft.fftfreq(self.slm_width, d=pixel_size)
        self.fy = np.fft.fftfreq(self.slm_height, d=pixel_size)
        self.FX, self.FY = np.meshgrid(self.fx, self.fy)

    def generate_kernel_phase(self, kernel_type='sharpen'):
        """
        生成常见卷积核的相位图
        """
        kernels = {
            'sharpen': np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]]),
            'edge_sobel_x': np.array([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]]),
            'edge_sobel_y': np.array([[-1, -2, -1],
                                      [0, 0, 0],
                                      [1, 2, 1]]),
            'laplacian': np.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]]),
            'gaussian': (1 / 273) * np.array([[1, 4, 7, 4, 1],
                                              [4, 16, 26, 16, 4],
                                              [7, 26, 41, 26, 7],
                                              [4, 16, 26, 16, 4],
                                              [1, 4, 7, 4, 1]]),
            'box_blur': np.ones((3, 3)) / 9
        }

        kernel = kernels.get(kernel_type, kernels['sharpen'])

        # 将核扩展到SLM分辨率
        kernel_padded = np.zeros((self.slm_height, self.slm_width))
        kh, kw = kernel.shape
        start_h = self.slm_height // 2 - kh // 2
        start_w = self.slm_width // 2 - kw // 2
        kernel_padded[start_h:start_h + kh, start_w:start_w + kw] = kernel

        # 计算傅里叶变换（滤波器函数）
        kernel_ft = fftshift(fft2(ifftshift(kernel_padded)))

        # 将滤波器转换为纯相位滤波器（仅保留相位信息）
        # 这是SLM可以实现的调制方式
        phase_filter = np.angle(kernel_ft)

        # 归一化到0-2π范围
        phase_map = (phase_filter - np.min(phase_filter)) % (2 * np.pi)

        return phase_map, kernel_padded

    def generate_phase_for_image(self, image, operation='sharpen'):
        """
        为特定图像和操作生成SLM相位图
        """
        # 1. 将图像转换为SLM分辨率
        img_resized = cv2.resize(image, (self.slm_width, self.slm_height))

        if len(img_resized.shape) == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized

        # 2. 计算图像的傅里叶变换
        img_ft = fftshift(fft2(img_gray))

        # 3. 获取卷积核的相位滤波器
        kernel_phase, _ = self.generate_kernel_phase(operation)

        # 4. 在傅里叶域应用滤波器
        # 注意：实际上我们需要在4f系统的傅里叶平面应用这个滤波器
        # 这里我们计算期望的输出相位
        img_phase = np.angle(img_ft)
        kernel_phase_resized = cv2.resize(kernel_phase, (self.slm_width, self.slm_height))

        # 5. 组合相位（模拟光学卷积）
        combined_phase = (img_phase + kernel_phase_resized) % (2 * np.pi)

        return combined_phase

    def generate_blazed_grating(self, period=100, direction='x'):
        """
        生成闪耀光栅相位图（用于光束偏转，消除零级光）
        period: 光栅周期（像素数）
        """
        if direction == 'x':
            phase = 2 * np.pi * (self.X % period) / period
        else:
            phase = 2 * np.pi * (self.Y % period) / period

        return phase % (2 * np.pi)

    def generate_fresnel_lens(self, focal_length=0.5):
        """
        生成菲涅尔透镜相位图（用于聚焦）
        focal_length: 焦距（米）
        """
        r_squared = (self.X * self.pixel_size) ** 2 + (self.Y * self.pixel_size) ** 2
        phase = (np.pi * r_squared) / (self.wavelength * focal_length)

        return phase % (2 * np.pi)

    def visualize_phase_map(self, phase_map, title="SLM Phase Map"):
        """
        可视化相位图
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(phase_map, cmap='hsv', extent=[-1, 1, -1, 1])
        plt.colorbar(label='Phase (rad)')
        plt.title(title)
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.show()

    def save_phase_as_image(self, phase_map, filename="slm_phase.bmp"):
        """
        将相位图保存为SLM可加载的图像格式
        相位值0-2π映射到灰度值0-255
        """
        # 定义保存目录
        save_dir = "SLM_imageProcessor"
        # 确保目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 构建完整保存路径
        save_path = os.path.join(save_dir, filename)
        # 归一化到0-255
        phase_normalized = (phase_map / (2 * np.pi) * 255).astype(np.uint8)

        # 保存为BMP格式（SLM软件支持）
        cv2.imwrite(save_path, phase_normalized)
        print(f"Phase map saved as {save_path}")

        return phase_normalized


# 使用示例
if __name__ == "__main__":
    # 初始化SLM处理器
    slm = SLM_ImageProcessor()

    # 示例1：生成锐化滤波器的相位图
    print("生成锐化滤波器相位图...")
    sharpen_phase, kernel = slm.generate_kernel_phase('sharpen')
    slm.visualize_phase_map(sharpen_phase, "Sharpen Filter Phase Map")
    slm.save_phase_as_image(sharpen_phase, "sharpen_phase.bmp")

    # 示例2：生成边缘检测滤波器相位图
    print("生成边缘检测滤波器相位图...")
    edge_phase, _ = slm.generate_kernel_phase('edge_sobel_x')
    slm.save_phase_as_image(edge_phase, "edge_detection_phase.bmp")

    # 示例3：生成闪耀光栅（用于消除零级光）
    print("生成闪耀光栅相位图...")
    grating_phase = slm.generate_blazed_grating(period=50)
    slm.save_phase_as_image(grating_phase, "blazed_grating.bmp")

    # 示例4：组合相位（锐化+闪耀光栅）
    print("生成组合相位图...")
    combined_phase = (sharpen_phase + 0.5 * grating_phase) % (2 * np.pi)
    slm.save_phase_as_image(combined_phase, "combined_phase.bmp")