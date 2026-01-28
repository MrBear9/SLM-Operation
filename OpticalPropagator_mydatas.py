import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import glob


class OpticalPropagator(nn.Module):
    """
    模拟光通过SLM的传播（角谱法）
    参考：用户手册中的全息计算原理
    """

    def __init__(self, wavelength=532e-9, pixel_size=8e-6, distance=0.1):
        super().__init__()
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distance = distance
        self.k = 2 * np.pi / wavelength

    def forward(self, phase_slm, input_field=None):
        """
        phase_slm: SLM相位调制 [batch, height, width]
        input_field: 输入光场（默认为平面波）
        """
        batch, H, W = phase_slm.shape

        # 如果没有输入光场，使用平面波
        if input_field is None:
            input_field = torch.ones(batch, H, W, dtype=torch.complex64, device=phase_slm.device)

        # 应用SLM相位调制
        modulated_field = input_field * torch.exp(1j * phase_slm)

        # 角谱法传播
        fx = torch.fft.fftfreq(W, d=self.pixel_size).to(phase_slm.device)
        fy = torch.fft.fftfreq(H, d=self.pixel_size).to(phase_slm.device)

        # 创建网格 - 兼容旧版本PyTorch的方法
        FX, FY = torch.meshgrid(fx, fy)
        # 转置以匹配 'xy' 索引
        FX = FX.t()
        FY = FY.t()

        # 传递函数
        k_squared = self.k ** 2
        freq_term = (2 * np.pi * FX) ** 2 + (2 * np.pi * FY) ** 2

        # 避免根号内出现负数
        sqrt_term = torch.sqrt(torch.relu(k_squared - freq_term))
        H_transfer = torch.exp(1j * self.distance * sqrt_term)

        # 傅里叶变换
        field_ft = torch.fft.fft2(modulated_field)

        # 应用传递函数
        field_ft_prop = field_ft * H_transfer.unsqueeze(0)

        # 逆傅里叶变换
        output_field = torch.fft.ifft2(field_ft_prop)

        return output_field


class SLM_PhaseNet(nn.Module):
    """
    神经网络：从输入图像和目标处理类型生成SLM相位图
    修改以适应1280x720输入
    """

    def __init__(self, input_channels=1, output_channels=1, input_size=(720, 1280)):
        super().__init__()
        self.input_size = input_size  # (height, width)

        # 计算编码器输出特征图大小
        # 经过两次2倍池化，所以尺寸除以4
        self.feat_height = input_size[0] // 4
        self.feat_width = input_size[1] // 4

        # 编码器 - 根据输入尺寸调整
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 操作类型编码器 - 动态适应特征图大小
        self.op_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, self.feat_height * self.feat_width),  # 动态计算大小
        )

        # 瓶颈层（处理类型编码）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 + 5, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )

        # 解码器（上采样到原始分辨率）
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 1),
            nn.Tanh(),  # 输出范围[-1, 1]，对应相位[-π, π]
        )

    def forward(self, x, operation_type):
        """
        x: 输入图像 [batch, 1, H, W]
        operation_type: 操作类型one-hot编码 [batch, 5]
        """
        # 编码图像特征
        feat = self.encoder(x)

        # 编码操作类型并广播到空间维度
        op_feat = self.op_encoder(operation_type)

        # 获取特征图的空间维度
        batch_size, channels, H_feat, W_feat = feat.shape

        # 重塑操作类型特征以匹配特征图的空间维度
        op_feat = op_feat.view(batch_size, 1, H_feat, W_feat)
        op_feat = op_feat.expand(-1, 5, -1, -1)  # 扩展到5个通道

        # 拼接特征
        combined = torch.cat([feat, op_feat], dim=1)

        # 通过瓶颈层
        bottleneck_out = self.bottleneck(combined)

        # 解码为相位图
        phase = self.decoder(bottleneck_out) * np.pi  # 缩放回[-π, π]

        return phase


class ImageProcessingDataset(Dataset):
    """
    数据集：输入图像 + 目标处理结果
    修改为适应1280x720图像
    """

    def __init__(self, image_dir, transform=None, target_size=(720, 1280)):
        """
        image_dir: 图像目录路径
        target_size: 目标图像大小 (height, width)
        """
        # 获取所有图像文件路径
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                           glob.glob(os.path.join(image_dir, "*.png")) + \
                           glob.glob(os.path.join(image_dir, "*.jpeg"))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        self.transform = transform
        self.target_size = target_size

        # 定义处理操作
        self.operations = ['sharpen', 'edge', 'blur', 'laplacian', 'gaussian']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # 加载图像
            img = cv2.imread(self.image_paths[idx])
            if img is None:
                print(f"Warning: Could not read image {self.image_paths[idx]}")
                # 返回一个替代图像
                img = np.random.randint(0, 255, (self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            else:
                # 调整大小到目标尺寸
                img = cv2.resize(img, (self.target_size[1], self.target_size[0]))

            # 转换为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0

            # 随机选择一种处理操作
            op_idx = np.random.randint(0, len(self.operations))
            op_type = self.operations[op_idx]

            # 生成目标处理结果
            target = self.apply_operation(img, op_type)

            # 转换为张量
            img_tensor = torch.FloatTensor(img).unsqueeze(0)  # [1, H, W]
            target_tensor = torch.FloatTensor(target).unsqueeze(0)

            # 操作类型one-hot编码
            op_onehot = torch.zeros(len(self.operations))
            op_onehot[op_idx] = 1

            return img_tensor, target_tensor, op_onehot

        except Exception as e:
            print(f"Error processing image {self.image_paths[idx]}: {e}")
            # 返回一个随机图像作为替代
            img = np.random.rand(self.target_size[0], self.target_size[1]).astype(np.float32)
            target = img.copy()

            op_onehot = torch.zeros(len(self.operations))
            op_onehot[0] = 1

            img_tensor = torch.FloatTensor(img).unsqueeze(0)
            target_tensor = torch.FloatTensor(target).unsqueeze(0)

            return img_tensor, target_tensor, op_onehot

    def apply_operation(self, img, operation):
        """应用数字图像处理操作"""
        if operation == 'sharpen':
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            result = cv2.filter2D(img, -1, kernel)
        elif operation == 'edge':
            result = cv2.Laplacian(img, cv2.CV_32F)
        elif operation == 'blur':
            result = cv2.GaussianBlur(img, (5, 5), 0)
        elif operation == 'laplacian':
            result = cv2.Laplacian(img, cv2.CV_32F)
        elif operation == 'gaussian':
            result = cv2.GaussianBlur(img, (5, 5), 1.0)
        else:
            result = img.copy()

        return np.clip(result, 0, 1)


def train_neural_phase_optimizer(image_dir, epochs=50, batch_size=4):
    """
    训练神经网络生成SLM相位图
    image_dir: 训练图像目录路径
    """
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置图像尺寸
    target_size = (720, 1280)  # height, width

    # 创建数据集
    dataset = ImageProcessingDataset(image_dir, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Dataset size: {len(dataset)} images")

    # 模型 - 使用新的输入尺寸
    phase_net = SLM_PhaseNet(input_size=target_size).to(device)
    propagator = OpticalPropagator().to(device)

    # 优化器
    optimizer = optim.Adam(phase_net.parameters(), lr=1e-4)

    # 损失函数
    mse_loss = nn.MSELoss()
    phase_smooth_loss = nn.L1Loss()

    # 训练循环
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (images, targets, ops) in enumerate(dataloader):
            # 移动数据到设备
            images = images.to(device)
            targets = targets.to(device)
            ops = ops.to(device)

            # 生成相位图
            phase_maps = phase_net(images, ops)

            # 去除通道维度，从 [batch, 1, H, W] 变为 [batch, H, W]
            phase_maps_2d = phase_maps.squeeze(1)

            # 光学传播
            output_fields = propagator(phase_maps_2d)
            output_intensity = torch.abs(output_fields) ** 2

            # 计算损失
            intensity_loss = mse_loss(output_intensity, targets)

            # 计算平滑损失
            smooth_loss = phase_smooth_loss(
                phase_maps_2d[:, 1:, :] - phase_maps_2d[:, :-1, :],
                torch.zeros_like(phase_maps_2d[:, 1:, :])
            )

            total_loss = intensity_loss + 0.1 * smooth_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            batch_count += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")

    return phase_net


def test_model(phase_net, test_image_path, operation_type_idx=0):
    """
    测试训练好的模型
    phase_net: 训练好的模型
    test_image_path: 测试图像路径
    operation_type_idx: 要应用的操作类型索引
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试图像
    img = cv2.imread(test_image_path)
    img = cv2.resize(img, (1280, 720))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0

    # 转换为张量
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

    # 创建操作类型one-hot编码
    op_onehot = torch.zeros(5)
    op_onehot[operation_type_idx] = 1
    op_onehot = op_onehot.unsqueeze(0).to(device)  # [1, 5]

    # 生成相位图
    phase_net.eval()
    with torch.no_grad():
        phase_map = phase_net(img_tensor, op_onehot)

    # 显示结果
    phase_map_np = phase_map.squeeze().cpu().numpy()

    # 保存相位图
    cv2.imwrite('phase_map.png', ((phase_map_np + np.pi) / (2 * np.pi) * 255).astype(np.uint8))
    print(f"Phase map saved as 'phase_map.png'")

    return phase_map_np


# 主程序
if __name__ == "__main__":
    # 设置你自己的图像目录路径
    image_dir = "images"  # 替换为你的图像目录路径

    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist. Using dummy data for demonstration.")

        # 如果没有真实数据，创建一些随机数据用于演示
        dummy_images = []
        for i in range(100):
            # 创建随机图像
            img = np.random.rand(720, 1280).astype(np.float32)
            dummy_images.append(img)

        # 训练模型
        print("训练神经网络相位优化器...")
        phase_net = train_neural_phase_optimizer(image_dir, epochs=20, batch_size=2)
        print("训练完成!")

        # 保存模型
        torch.save(phase_net.state_dict(), 'phase_net_model.pth')
        print("模型已保存为 'phase_net_model.pth'")
    else:
        # 使用真实数据训练
        print("训练神经网络相位优化器...")
        phase_net = train_neural_phase_optimizer(image_dir, epochs=50, batch_size=4)
        print("训练完成!")

        # 保存模型
        torch.save(phase_net.state_dict(), 'phase_net_model.pth')
        print("模型已保存为 'phase_net_model.pth'")

        # 测试模型（如果有测试图像）
        test_image_path = "images/zidane.jpg"  # 替换为你的测试图像路径
        if os.path.exists(test_image_path):
            phase_map = test_model(phase_net, test_image_path, operation_type_idx=0)