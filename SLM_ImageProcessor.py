import numpy as np
import cv2,os
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift

# 定义保存目录
save_dir = "SLM_imageProcessor_test_768x768"
# 确保目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class SLM_ImageProcessor:
    def __init__(self, slm_resolution=(768, 768), wavelength=532e-9, pixel_size=8e-6):
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

    def load_image(self, image_path):
        """
        加载图像，支持多种格式

        参数:
            image_path: 图像路径或numpy数组

        返回:
            加载的图像
        """
        if isinstance(image_path, str):
            # 如果是字符串路径，从文件加载
            if not os.path.exists(image_path):
                # 尝试添加扩展名
                possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                found = False
                for ext in possible_extensions:
                    if os.path.exists(image_path + ext):
                        image_path = image_path + ext
                        found = True
                        break

                if not found:
                    raise FileNotFoundError(f"图像文件未找到: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")

            # 转换为RGB格式（如果原本是BGR）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
        elif isinstance(image_path, np.ndarray):
            # 如果已经是numpy数组，直接返回
            return image_path
        else:
            raise TypeError("image_path必须是字符串路径或numpy数组")

    def generate_phase_for_image(self, image_path='images/zidane.jpg', operation='sharpen',
                                 save_result=True, show_result=True):
        """
        为指定图像生成卷积操作的相位图
        基于4f光学卷积系统原理

        参数:
            image_path: 图像路径或numpy数组
            operation: 卷积操作类型 ('sharpen', 'edge_sobel_x', 'edge_sobel_y',
                       'laplacian', 'gaussian', 'box_blur')
            save_result: 是否保存结果
            show_result: 是否显示结果

        返回:
            phase_map: 生成的相位图
            processed_image: 处理后的图像（模拟结果）
        """
        # 1. 加载图像
        try:
            image = self.load_image(image_path)
        except Exception as e:
            print(f"图像加载失败: {e}")
            # 创建一个默认测试图像
            print("使用默认测试图像...")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(image, 'Test Image', (150, 256),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # 2. 将图像转换为灰度图并调整大小
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        # 调整到SLM分辨率
        image_resized = cv2.resize(image_gray, (self.slm_width, self.slm_height))

        # 3. 获取卷积核
        kernel_phase, kernel = self.generate_kernel_phase(operation)
        print(kernel.shape)

        # 4. 模拟4f系统的光学卷积
        # 4.1 输入图像傅里叶变换
        image_ft = fftshift(fft2(image_resized))

        # 4.2 应用相位滤波器（模拟SLM在傅里叶平面调制）
        # 注意：在纯相位调制中，我们只改变相位，保持振幅不变
        filtered_ft = image_ft * np.exp(1j * kernel_phase)

        # 4.3 逆傅里叶变换得到输出图像
        output_image = np.abs(ifft2(ifftshift(filtered_ft)))

        # 归一化输出图像
        output_image = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX)
        output_image = output_image.astype(np.uint8)

        # 5. 为SLM生成相位图
        # 在4f系统中，SLM位于傅里叶平面，加载的是卷积核的相位
        phase_map = kernel_phase

        # 6. 保存和显示结果
        if save_result:
            # 保存相位图
            phase_filename = f"image_{operation}_phase.bmp"
            self.save_phase_as_image(phase_map, phase_filename)
            # self.save_phase_as_image(kernel,"kerneltestimage.bmp")

            # 保存原始图像和处理后的图像
            original_filename = f"original_{os.path.basename(image_path).split('.')[0]}.png"
            processed_filename = f"processed_{operation}_{os.path.basename(image_path).split('.')[0]}.png"

            cv2.imwrite(os.path.join(save_dir, original_filename), image_resized)
            cv2.imwrite(os.path.join(save_dir, processed_filename), output_image)

            print(f"原始图像保存为: {original_filename}")
            print(f"处理后的图像保存为: {processed_filename}")

        # if show_result:
        #     # 显示结果对比
        #     self._display_image_comparison(image_resized, output_image, operation)

        return phase_map, output_image

    # def _display_image_comparison(self, original, processed, operation):
    #     """
    #     显示原始图像和处理后图像的对比
    #
    #     参数:
    #         original: 原始图像
    #         processed: 处理后的图像
    #         operation: 操作名称
    #     """
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    #     # 显示原始图像
    #     axes[0].imshow(original, cmap='gray')
    #     axes[0].set_title('Original Image')
    #     axes[0].axis('off')
    #
    #     # 显示处理后的图像
    #     axes[1].imshow(processed, cmap='gray')
    #     axes[1].set_title(f'After {operation} Filter')
    #     axes[1].axis('off')
    #
    #     # 显示相位图（可选）
    #     # 获取相位图
    #     kernel_phase, _ = self.generate_kernel_phase(operation)
    #     axes[2].imshow(kernel_phase, cmap='hsv')
    #     axes[2].set_title(f'{operation} Phase Map')
    #     axes[2].axis('off')
    #
    #     plt.tight_layout()
    #     plt.show()

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

    def generate_vortex_beam(self, topological_charge=1, radius_factor=0.8):
        """
        生成涡旋光束相位图

        参数:
            topological_charge: 拓扑电荷数，决定相位缠绕次数
            radius_factor: 涡旋光束的有效半径因子（0-1）

        返回:
            涡旋光束相位图 (0-2π)
        """
        # 计算方位角（极坐标中的角度）
        # 注意：np.arctan2(Y, X) 返回角度范围为 [-π, π]
        theta = np.arctan2(self.Y, self.X)

        # 将角度范围转换到 [0, 2π]
        theta = np.mod(theta, 2 * np.pi)

        # 生成涡旋相位：l * θ
        vortex_phase = topological_charge * theta

        # 应用圆形孔径掩模
        r = np.sqrt(self.X ** 2 + self.Y ** 2)
        max_radius = radius_factor * min(self.slm_width, self.slm_height) / 2
        aperture_mask = r <= max_radius

        # 在掩模外相位为0
        vortex_phase = vortex_phase * aperture_mask

        # 归一化到0-2π范围
        vortex_phase = np.mod(vortex_phase, 2 * np.pi)

        return vortex_phase

    def generate_dammann_grating(self, period_x=50, period_y=50, duty_cycle=0.5, phase_values=(0, np.pi)):
        """
        生成达曼光栅相位图（二值相位光栅）

        参数:
            period_x: x方向周期（像素数）
            period_y: y方向周期（像素数）
            duty_cycle: 占空比（高相位部分占比，0-1）
            phase_values: (相位1, 相位2) 两个相位值，通常为(0, π)

        返回:
            达曼光栅相位图
        """
        # 创建周期性相位图
        # 使用正弦函数创建周期性变化，然后二值化
        grating_x = (self.X / period_x) % 1.0
        grating_y = (self.Y / period_y) % 1.0

        # 创建二维达曼光栅
        dammann_grating = np.zeros_like(self.X)

        # 生成二值相位图案
        # 当grating_x和grating_y都在duty_cycle范围内时，使用phase1，否则使用phase2
        mask_x = grating_x < duty_cycle
        mask_y = grating_y < duty_cycle

        # 二维达曼光栅可以是各种设计，这里实现一个简单的棋盘格设计
        dammann_grating = np.where(
            (mask_x & mask_y) | ((~mask_x) & (~mask_y)),
            phase_values[0],
            phase_values[1]
        )

        # 归一化到0-2π范围
        dammann_grating = np.mod(dammann_grating, 2 * np.pi)

        return dammann_grating

    def generate_vortex_dammann_grating(self, topological_charge=1,
                                        dammann_period_x=50, dammann_period_y=50,
                                        vortex_radius_factor=0.8):
        """
        生成涡旋光束与达曼光栅的叠加相位图
        用于生成多焦点涡旋光束阵列

        参数:
            topological_charge: 涡旋拓扑电荷数
            dammann_period_x: 达曼光栅x方向周期
            dammann_period_y: 达曼光栅y方向周期
            vortex_radius_factor: 涡旋光束有效半径因子

        返回:
            叠加相位图
        """
        # 生成涡旋光束相位
        vortex_phase = self.generate_vortex_beam(
            topological_charge=topological_charge,
            radius_factor=vortex_radius_factor
        )

        # 生成达曼光栅相位
        dammann_phase = self.generate_dammann_grating(
            period_x=dammann_period_x,
            period_y=dammann_period_y,
            duty_cycle=0.5,
            phase_values=(0, np.pi)
        )

        # 叠加相位（取模2π）
        combined_phase = np.mod(vortex_phase + dammann_phase, 2 * np.pi)

        return combined_phase, vortex_phase, dammann_phase

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

    def visualize_multiple_phases(self, phase_maps, titles=None, save_path=None):
        """
        同时可视化多个相位图

        参数:
            phase_maps: 相位图列表
            titles: 对应的标题列表
            save_path: 保存路径（可选）
        """
        n_plots = len(phase_maps)

        if titles is None:
            titles = [f"Phase Map {i + 1}" for i in range(n_plots)]

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

        if n_plots == 1:
            axes = [axes]

        for i, (phase_map, title) in enumerate(zip(phase_maps, titles)):
            ax = axes[i]
            im = ax.imshow(phase_map, cmap='hsv', extent=[-1, 1, -1, 1])
            ax.set_title(title)
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            plt.colorbar(im, ax=ax, label='Phase (rad)')

        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_dir, save_path), dpi=150)
            print(f"Figure saved as {save_path}")

        plt.show()

    def save_phase_as_image(self, phase_map, filename="slm_phase.bmp"):
        """
        将相位图保存为SLM可加载的图像格式
        相位值0-2π映射到灰度值0-255
        """
        # 构建完整保存路径
        save_path = os.path.join(save_dir, filename)
        # 归一化到0-255
        phase_normalized = (phase_map / (2 * np.pi) * 255).astype(np.uint8)

        # 保存为BMP格式（SLM软件支持）
        cv2.imwrite(save_path, phase_normalized)
        print(f"Phase map saved as {save_path}")

        return phase_normalized


class SLM_Image_shift():
    """
    图像处理演示类
    使用SLM_ImageProcessor为图像生成相位图
    """

    def __init__(self, image_path='images/zidane.jpg'):
        """
        初始化

        参数:
            image_path: 图像路径，默认为'images/zidane.jpg'
        """
        self.image_path = image_path
        self.slm = SLM_ImageProcessor()

    def process_image(self, operation='sharpen'):
        """
        处理图像并生成相位图

        参数:
            operation: 卷积操作类型

        返回:
            phase_map: 生成的相位图
            processed_image: 处理后的图像
        """
        print(f"处理图像: {self.image_path}")
        print(f"使用操作: {operation}")

        # 生成相位图
        phase_map, processed_image = self.slm.generate_phase_for_image(
            image_path=self.image_path,
            operation=operation,
            save_result=True,
            show_result=True
        )

        # 可视化相位图
        self.slm.visualize_phase_map(phase_map, f"{operation} Filter Phase Map")

        # # 保存相位图
        # self.slm.save_phase_as_image(phase_map, f"{operation}_phase.bmp")

        return phase_map, processed_image

    def demo_all_operations(self):
        """
        演示所有可用的卷积操作
        """
        operations = ['sharpen', 'edge_sobel_x', 'edge_sobel_y',
                      'laplacian', 'gaussian', 'box_blur']

        print("=" * 60)
        print("演示所有卷积操作")
        print("=" * 60)

        results = {}
        for op in operations:
            print(f"\n处理操作: {op}")
            try:
                phase, processed = self.process_image(op)
                results[op] = (phase, processed)
            except Exception as e:
                print(f"操作 {op} 失败: {e}")

        print("\n所有操作完成!")
        return results

class SLM_CommonGenerate():
    # 初始化SLM处理器
    slm = SLM_ImageProcessor()

    # 使用SLM_Image_shift处理图像
    print("使用SLM_Image_shift处理图像...")
    image_processor = SLM_Image_shift('images/004355.jpg')

    # 处理单个操作
    print("处理操作...")
    sharpen_phase, sharpen_image = image_processor.process_image('edge_sobel_y')

    # 或者演示所有操作
    # image_processor.demo_all_operations()

    # print("生成菲涅尔透镜相位图...")
    # sharpen_phase = slm.generate_fresnel_lens(0.5)
    # slm.visualize_phase_map(sharpen_phase, "fresnel_lens Filter Phase Map")
    # slm.save_phase_as_image(sharpen_phase, "fresnel_Phase.bmp")
    #
    # # 生成涡旋光束相位图
    # print("生成涡旋光束相位图...")
    # vortex_phase = slm.generate_vortex_beam(topological_charge=2, radius_factor=0.7)
    # slm.save_phase_as_image(vortex_phase, "vortex_beam_phase.bmp")
    #
    # # 生成达曼光栅相位图
    # print("生成达曼光栅相位图...")
    # dammann_phase = slm.generate_dammann_grating(period_x=60, period_y=60)
    # slm.save_phase_as_image(dammann_phase, "dammann_grating_phase.bmp")
    #
    # # 生成涡旋达曼光栅叠加相位图
    # print("生成涡旋达曼光栅叠加相位图...")
    # combined_phase, vortex_only, dammann_only = slm.generate_vortex_dammann_grating(
    #     topological_charge=3,
    #     dammann_period_x=80,
    #     dammann_period_y=80,
    #     vortex_radius_factor=0.6
    # )
    #
    # # 可视化所有相位图
    # slm.visualize_multiple_phases(
    #     [vortex_only, dammann_only, combined_phase],
    #     ["Vortex Beam (l=3)", "Dammann Grating", "Combined Vortex-Dammann"],
    #     save_path="vortex_dammann_comparison.png"
    # )
    #
    # # 保存叠加相位图
    # slm.save_phase_as_image(combined_phase, "vortex_dammann_combined_phase.bmp")
    #
    # # 组合涡旋达曼与闪耀光栅（用于消除零级光）
    # print("生成涡旋达曼+闪耀光栅相位图...")
    # blazed_phase = slm.generate_blazed_grating(period=50)
    # final_phase = np.mod(combined_phase + 0.3 * blazed_phase, 2 * np.pi)
    # slm.save_phase_as_image(final_phase, "vortex_dammann_blazed_phase.bmp")

    # # 示例1：生成锐化滤波器的相位图
    # print("生成锐化滤波器相位图...")
    # sharpen_phase, kernel = slm.generate_kernel_phase('sharpen')
    # slm.visualize_phase_map(sharpen_phase, "Sharpen Filter Phase Map")
    # slm.save_phase_as_image(sharpen_phase, "sharpen_phase.bmp")
    #
    # # 示例2：生成边缘检测滤波器相位图
    # print("生成边缘检测滤波器相位图...")
    # edge_phase, _ = slm.generate_kernel_phase('edge_sobel_x')
    # slm.save_phase_as_image(edge_phase, "edge_detection_phase.bmp")
    #
    # # 示例3：生成闪耀光栅（用于消除零级光）
    # print("生成闪耀光栅相位图...")
    # grating_phase = slm.generate_blazed_grating(period=50)
    # slm.save_phase_as_image(grating_phase, "blazed_grating.bmp")
    #
    # # 示例4：组合相位（锐化+闪耀光栅）
    # print("生成组合相位图...")
    # combined_phase = (sharpen_phase + 0.5 * grating_phase) % (2 * np.pi)
    # slm.save_phase_as_image(combined_phase, "combined_phase.bmp")


if __name__ == "__main__":
    SLM_CommonGenerate()
