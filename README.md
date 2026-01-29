# 系统结构

典型的4f系统由以下部分组成：

~~~txt
输入平面 → 透镜1 (f) → 傅里叶平面 (SLM) → 透镜2 (f) → 输出平面
     f          f            f          f
~~~

总长度为4倍焦距，因此称为"4f系统"。

# 数学模型

## 1.输入信号

输入图像函数(image_resized)：
$$
g_{in}(x_1, y_1) \in \mathbb{R}^{M \times N}
$$

## 2. 第一个透镜的傅里叶变换

第一个透镜对输入图像进行傅里叶变换：
$$
G(u, v) = \mathcal{F}\{g_{in}(x_1, y_1)\} = \iint g_{in}(x_1, y_1) \cdot e^{-j2\pi(ux_1 + vy_1)} dx_1 dy_1
$$
代码实现：

~~~python
image_ft = fftshift(fft2(image_resized))  # G(u,v)
~~~

## 3. 傅里叶平面上的相位调制

在傅里叶平面（频谱面），SLM施加相位调制：
$$
H_{phase}(u,v)=e^{jϕ(u,v)}
$$
其中$ϕ(u,v)$是卷积核的相位。

代码中的调制过程：

~~~python
filtered_ft = image_ft * np.exp(1j * kernel_phase)  # G(u,v) * H_phase(u,v)
~~~

## 4. 光学卷积原理

理论上，卷积操作在频域表示为乘法：
$$
G_{out}(u,v) = G_{in}(u,v) \cdot H(u,v)
$$
其中$H(u,v)=\mathcal{F}\{h(x,y)\}$是卷积核的傅里叶变换。

在纯相位调制系统中，我们只有相位信息：
$$
H_{phase}(u,v) = \frac{H(u,v)}{|H(u,v)|} = e^{j \cdot \text{angle}(H(u,v))}
$$

## 5. 第二个透镜的逆傅里叶变换

第二个透镜进行逆傅里叶变换，得到输出：
$$
g_{out}(x_2, y_2) = \mathcal{F}^{-1}\{G_{out}(u,v)\} = |\mathcal{F}^{-1}\{G_{in}(u,v) \cdot H_{phase}(u,v)\}|
$$
代码实现：

~~~python
output_image = np.abs(ifft2(ifftshift(filtered_ft)))
~~~

## **完整的系统公式**

4f系统的完整数学描述为：
$$
g_{out}(x_2, y_2) = |\mathcal{F}^{-1} \{ \mathcal{F}\{g_{in}(x_1, y_1)\} \cdot e^{j\phi(u,v)} \}|
$$
其中：

- $ϕ(u,v)=angle(\mathcal{F}\{h(x,y)\})$
- $h(x,y)$是空间域的卷积核（如Sobel、Gaussian等）
- $g_{in}(x_1,y_1)$为输入图像--->image_resized
- $\mathcal{F}\{g_{in}\}$是第一次傅里叶变换--->fft2(image_resized)
- $*e^{jϕ(u,v)}$是相位滤波--->\* np.exp(1j*kernel_phase)
- $\mathcal{F}^{-1}\{·\}$为第二次傅里叶变换--->ifft2(filtered_ft)
- $g_{out}(x_2, y_2)$输出图像--->np.abs(output_image)

## 特点与限制

1. **纯相位调制**：只使用相位信息，丢失了振幅信息
2. **线性系统**：满足线性叠加原理
3. **空间不变性**：假设系统是空间不变的
4. **单色光假设**：通常使用单色相干光照明

这种光学卷积系统的优势在于**并行处理**和**高速计算**，所有像素点同时参与计算，适合实时图像处理任务。













