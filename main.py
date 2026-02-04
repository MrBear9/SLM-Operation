import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

# 创建保存图片的目录
output_dir = './images_kernels_768x768/'
os.makedirs(output_dir, exist_ok=True)

# 读取图像
image = cv2.imread('./images/004355.jpg')  # 请替换为您的图像路径
if image is None:
    # 如果没找到图像，创建一个示例图像
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(image, (100, 100), 30, (0, 0, 255), -1)
image = cv2.resize(image, (768, 768))

# 转换为灰度图并保存
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_dir, '00_original_grayscale.jpg'), gray)

# 1. 平滑/模糊核
# 均值模糊
blur_kernel = np.ones((5, 5), np.float32) / 25
blurred = cv2.filter2D(gray, -1, blur_kernel)
cv2.imwrite(os.path.join(output_dir, '01_blurred_mean.jpg'), blurred)

# 高斯模糊
gaussian_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite(os.path.join(output_dir, '02_gaussian_blurred.jpg'), gaussian_blurred)

# 2. 锐化核
# 方法1：基本锐化核
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
cv2.imwrite(os.path.join(output_dir, '03_sharpened_basic.jpg'), sharpened)

# 方法2：拉普拉斯锐化
laplacian_kernel = np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]])
laplacian = cv2.filter2D(gray, cv2.CV_64F, laplacian_kernel)
laplacian_sharpened = cv2.convertScaleAbs(gray - laplacian)
cv2.imwrite(os.path.join(output_dir, '04_laplacian_sharpened.jpg'), laplacian_sharpened)

# 方法3：非锐化掩蔽（Unsharp Masking）
gaussian = cv2.GaussianBlur(gray, (5, 5), 2.0)
unsharp_masked = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
cv2.imwrite(os.path.join(output_dir, '05_unsharp_masked.jpg'), unsharp_masked)

# 3. 边缘检测核
# Sobel算子（水平方向）
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
sobel_x = cv2.filter2D(gray, cv2.CV_64F, sobel_x_kernel)
sobel_x = np.uint8(np.absolute(sobel_x))
cv2.imwrite(os.path.join(output_dir, '06_sobel_x.jpg'), sobel_x)

# Sobel算子（垂直方向）
sobel_y_kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
sobel_y = cv2.filter2D(gray, cv2.CV_64F, sobel_y_kernel)
sobel_y = np.uint8(np.absolute(sobel_y))
cv2.imwrite(os.path.join(output_dir, '07_sobel_y.jpg'), sobel_y)

# Sobel组合（梯度幅值）
sobel_combined = cv2.magnitude(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
                               cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
sobel_combined = np.uint8(sobel_combined)
cv2.imwrite(os.path.join(output_dir, '08_sobel_combined.jpg'), sobel_combined)

# Prewitt算子
prewitt_x_kernel = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
prewitt_x = cv2.filter2D(gray, cv2.CV_64F, prewitt_x_kernel)
prewitt_x = np.uint8(np.absolute(prewitt_x))
cv2.imwrite(os.path.join(output_dir, '09_prewitt_x.jpg'), prewitt_x)

# Roberts算子
roberts_x_kernel = np.array([[1, 0],
                             [0, -1]])
roberts_y_kernel = np.array([[0, 1],
                             [-1, 0]])
roberts_x = cv2.filter2D(gray, cv2.CV_64F, roberts_x_kernel)
roberts_y = cv2.filter2D(gray, cv2.CV_64F, roberts_y_kernel)
roberts = np.uint8(np.sqrt(roberts_x ** 2 + roberts_y ** 2))
cv2.imwrite(os.path.join(output_dir, '10_roberts.jpg'), roberts)

# Canny边缘检测（虽然不是卷积核，但常用）
canny_edges = cv2.Canny(gray, 100, 200)
cv2.imwrite(os.path.join(output_dir, '11_canny_edges.jpg'), canny_edges)

# 4. 形态学操作核
# 膨胀
dilation_kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(gray, dilation_kernel, iterations=1)
cv2.imwrite(os.path.join(output_dir, '12_dilated.jpg'), dilated)

# 腐蚀
erosion_kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(gray, erosion_kernel, iterations=1)
cv2.imwrite(os.path.join(output_dir, '13_eroded.jpg'), eroded)

# 5. 梯度核（用于边缘增强）
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
cv2.imwrite(os.path.join(output_dir, '14_gradient.jpg'), gradient)

# 6. 自定义卷积核示例
# 浮雕效果
emboss_kernel = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])
emboss = cv2.filter2D(gray, -1, emboss_kernel)
cv2.imwrite(os.path.join(output_dir, '15_emboss.jpg'), emboss)

# 轮廓增强
outline_kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
outline = cv2.filter2D(gray, -1, outline_kernel)
cv2.imwrite(os.path.join(output_dir, '16_outline.jpg'), outline)

# 7. 使用SciPy的卷积函数
from scipy.ndimage import convolve

# 创建自定义边缘检测核
custom_edge_kernel = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]])
scipy_convolved = convolve(gray.astype(float), custom_edge_kernel, mode='reflect')
scipy_convolved = np.uint8(np.clip(np.abs(scipy_convolved), 0, 255))
cv2.imwrite(os.path.join(output_dir, '17_scipy_convolved.jpg'), scipy_convolved)

# 可视化结果
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
images = [
    ('Original', gray),
    ('Blurred (5x5)', blurred),
    ('Gaussian Blur', gaussian_blurred),
    ('Sharpened', sharpened),
    ('Laplacian Sharpened', laplacian_sharpened),
    ('Unsharp Masking', unsharp_masked),
    ('Sobel X', sobel_x),
    ('Sobel Y', sobel_y),
    ('Sobel Combined', sobel_combined),
    ('Prewitt X', prewitt_x),
    ('Roberts', roberts),
    ('Canny Edges', canny_edges),
    ('Dilated', dilated),
    ('Eroded', eroded),
    ('Gradient', gradient),
    ('Emboss', emboss),
    ('Outline', outline),
    ('Scipy Convolved', scipy_convolved)
]

# 填充图像列表以确保网格完整
while len(images) < 20:
    images.append(('', np.zeros_like(gray)))

for idx, (title, img) in enumerate(images):
    ax = axes[idx // 5, idx % 5]
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '18_all_results_grid.png'), dpi=100, bbox_inches='tight')
plt.show()

# 打印各个核的信息
print("=" * 50)
print("卷积核详解：")
print("=" * 50)
print("\n1. 平滑核（均值模糊）：")
print(blur_kernel)
print("\n作用：降低图像噪声，模糊细节")

print("\n2. 锐化核：")
print(sharpen_kernel)
print("\n原理：增强中心像素，减弱周围像素，突出边缘")

print("\n3. Sobel X 核（水平边缘检测）：")
print(sobel_x_kernel)
print("\n原理：对水平方向梯度敏感，检测垂直边缘")

print("\n4. Sobel Y 核（垂直边缘检测）：")
print(sobel_y_kernel)
print("\n原理：对垂直方向梯度敏感，检测水平边缘")

print("\n5. Laplacian 核：")
print(laplacian_kernel)
print("\n原理：二阶微分算子，对边缘和角点敏感")

print("\n6. 浮雕核：")
print(emboss_kernel)
print("\n原理：创建3D浮雕效果")


# 手动卷积演示函数
def manual_convolution(image, kernel):
    """
    手动实现卷积操作
    """
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # 添加填充
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros_like(image, dtype=float)

    # 执行卷积
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kernel_h, j:j + kernel_w]
            result[i, j] = np.sum(region * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


# 测试手动卷积
print("\n手动卷积演示：")
test_kernel = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

manual_result = manual_convolution(gray, test_kernel)
opencv_result = cv2.filter2D(gray, -1, test_kernel)

print(f"手动卷积与OpenCV结果差异：{np.sum(np.abs(manual_result - opencv_result))}")

# 保存手动卷积结果
cv2.imwrite(os.path.join(output_dir, '19_manual_convolution.jpg'), manual_result)
cv2.imwrite(os.path.join(output_dir, '20_opencv_convolution.jpg'), opencv_result)

print(f"\n所有图片已保存到：{output_dir}")