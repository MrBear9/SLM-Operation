import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2


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

        # 操作类型编码器
        self.op_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64 * 64),  # 输出大小为64x64，匹配编码器输出特征图大小
        )

        # 瓶颈层（处理类型编码）
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 + 5, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )

        # 解码器（上采样到SLM分辨率）
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


def combine_phase_maps(phase_tensor, method='mean', filename='phase_map_combined.png'):
    """
    将多个相位通道组合成单一相位图
    """
    phase_np = phase_tensor.detach().cpu().numpy()

    if phase_np.ndim == 3:
        if method == 'mean':
            # 取平均值
            combined = np.mean(phase_np, axis=0)
        elif method == 'first':
            # 取第一个通道
            combined = phase_np[0]
        elif method == 'sum':
            # 求和
            combined = np.sum(phase_np, axis=0)
        elif method == 'principal':
            # 取主相位（幅度最大）
            # 假设前两个通道是实部和虚部
            if phase_np.shape[0] >= 2:
                real = phase_np[0]
                imag = phase_np[1]
                combined = np.arctan2(imag, real)
            else:
                combined = phase_np[0]
        else:
            combined = phase_np[0]

        # 归一化到 [0, 255]
        phase_norm = ((combined + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

        # 保存
        cv2.imwrite(filename, phase_norm)
        print(f"Combined phase map ({method}) saved as '{filename}'")

        return combined
    else:
        print(f"Cannot combine phase from shape {phase_np.shape}")
        return None




def train_neural_phase_optimizer():
    """
    训练神经网络生成SLM相位图
    """
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备：", device)
    # 模型
    phase_net = SLM_PhaseNet().to(device)
    propagator = OpticalPropagator().to(device)

    # 优化器
    optimizer = optim.Adam(phase_net.parameters(), lr=1e-4)

    # 损失函数
    mse_loss = nn.MSELoss()
    phase_smooth_loss = nn.L1Loss()

    # 训练循环
    for epoch in range(11):
        # 这里简化了数据加载，实际需要真实数据集
        batch_size = 4
        dummy_images = torch.randn(batch_size, 1, 256, 256).to(device)
        dummy_targets = torch.randn(batch_size, 1, 256, 256).to(device)
        dummy_ops = torch.eye(5)[torch.randint(0, 5, (batch_size,))].to(device)

        # 生成相位图
        phase_maps = phase_net(dummy_images, dummy_ops)

        # 去除通道维度，从 [batch, 1, H, W] 变为 [batch, H, W]
        phase_maps_2d = phase_maps.squeeze(1)

        # 光学传播
        output_fields = propagator(phase_maps_2d)
        output_intensity = torch.abs(output_fields) ** 2

        # 计算损失
        intensity_loss = mse_loss(output_intensity, dummy_targets)

        # 计算平滑损失时也使用去除通道维度的相位图
        smooth_loss = phase_smooth_loss(
            phase_maps_2d[:, 1:, :] - phase_maps_2d[:, :-1, :],
            torch.zeros_like(phase_maps_2d[:, 1:, :])
        )

        total_loss = intensity_loss + 0.1 * smooth_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
            # 显示结果
            phase_map_np = phase_maps_2d.detach().cpu().numpy()
            print(f"phase_map_np shape: {phase_map_np.shape}")  # 添加这行
            # 保存相位图
            # 使用
            combined_phase = combine_phase_maps(phase_maps_2d, method='mean')

    return phase_net


# 训练神经网络相位优化器
print("训练神经网络相位优化器...")
phase_net = train_neural_phase_optimizer()
print(phase_net)
