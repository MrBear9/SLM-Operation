import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


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
            input_field = torch.ones(batch, H, W, dtype=torch.complex64)

        # 应用SLM相位调制
        modulated_field = input_field * torch.exp(1j * phase_slm)

        # 角谱法传播
        fx = torch.fft.fftfreq(W, d=self.pixel_size).to(phase_slm.device)
        fy = torch.fft.fftfreq(H, d=self.pixel_size).to(phase_slm.device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')

        # 传递函数
        H_transfer = torch.exp(1j * self.distance *
                               torch.sqrt(self.k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2))

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
    """

    def __init__(self, input_channels=1, output_channels=1, slm_res=(1920, 1080)):
        super().__init__()

        # 编码器
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

        # 瓶颈层（处理类型编码）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 + 5, 512, 3, padding=1),  # +5 for operation type
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )

        # 解码器（上采样到SLM分辨率）
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 1),
            nn.Tanh(),  # 输出范围[-1, 1]，对应相位[-π, π]
        )

        # 操作类型编码器
        self.op_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, slm_res[0] // 4 * slm_res[1] // 4),
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
        op_feat = op_feat.view(feat.shape[0], 1, feat.shape[2], feat.shape[3])
        op_feat = op_feat.expand(-1, 5, -1, -1)  # 扩展到5个通道

        # 拼接特征
        combined = torch.cat([feat, op_feat], dim=1)

        # 解码为相位图
        phase = self.decoder(combined) * np.pi  # 缩放回[-π, π]

        return phase


class ImageProcessingDataset(Dataset):
    """
    数据集：输入图像 + 目标处理结果
    """

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

        # 定义处理操作
        self.operations = ['sharpen', 'edge', 'blur', 'laplacian', 'gaussian']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))
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


def train_neural_phase_optimizer():
    """
    训练神经网络生成SLM相位图
    """
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型
    phase_net = SLM_PhaseNet().to(device)
    propagator = OpticalPropagator().to(device)

    # 优化器
    optimizer = optim.Adam(phase_net.parameters(), lr=1e-4)

    # 损失函数
    mse_loss = nn.MSELoss()
    phase_smooth_loss = nn.L1Loss()

    # 训练循环
    for epoch in range(100):
        # 这里简化了数据加载，实际需要真实数据集
        batch_size = 4
        dummy_images = torch.randn(batch_size, 1, 256, 256).to(device)
        dummy_targets = torch.randn(batch_size, 1, 256, 256).to(device)
        dummy_ops = torch.eye(5)[torch.randint(0, 5, (batch_size,))].to(device)

        # 生成相位图
        phase_maps = phase_net(dummy_images, dummy_ops)

        # 光学传播
        output_fields = propagator(phase_maps)
        output_intensity = torch.abs(output_fields) ** 2

        # 计算损失
        intensity_loss = mse_loss(output_intensity, dummy_targets)
        smooth_loss = phase_smooth_loss(
            phase_maps[:, :, 1:, :] - phase_maps[:, :, :-1, :],
            torch.zeros_like(phase_maps[:, :, 1:, :])
        )

        total_loss = intensity_loss + 0.1 * smooth_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

    return phase_net


# 训练神经网络相位优化器
print("训练神经网络相位优化器...")
# phase_net = train_neural_phase_optimizer()