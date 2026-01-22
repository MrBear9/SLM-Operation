# SLM Operation - Spatial Light Modulator Image Processing Project

## Project Introduction

SLMOperation is an optical image processing project based on the Spatial Light Modulator (SLM). This project leverages the PyTorch deep learning framework to implement various optical convolution operations on images, including image sharpening, edge detection, and blurring. By integrating optical propagation principles with neural network techniques, the project transforms traditional digital image processing into an optical processing approach.

## Features

### 1. Optical Propagation Simulation
- **OpticalPropagator class**: Simulates the propagation of light waves in space, supporting Fresnel diffraction propagation
- Supports customizable parameters: wavelength, pixel size, and propagation distance
- Converts SLM phase maps into optical field distributions

### 2. SLM Phase Generation
- **SLM_ImageProcessor class**: Generates SLM phase maps required for various image processing operations
- Supports multiple convolution kernels: sharpen, edge detection, blur, laplacian, gaussian, etc.
- Provides functionality to generate blazed gratings and Fresnel lens phase patterns
- Includes visualization and saving capabilities for phase maps

### 3. Neural Network-Based Phase Optimization
- **SLM_PhaseNet class**: Neural network for optimizing SLM phase patterns
- Supports configurable input and output channels
- Enables training of optimized phase patterns

### 4. Dataset Processing
- **ImageProcessingDataset class**: Data loader for image processing tasks
- Supports image augmentation and multiple operation types

## File Structure

```
SLMOperation/
├── OpticalPropagator.py          # Optical propagation and neural network phase models
├── SLM_ImageProcessor.py         # Main class for SLM image processing
├── create_slm_phase_for_convolution.py  # Generate SLM convolution phase maps
├── main.py                       # Main program entry point
├── images/                       # Test image directory
│   ├── bus.jpg
│   └── zidane.jpg
├── SLM_imageProcessor/           # Example generated phase maps
│   ├── blazed_grating.bmp
│   ├── combined_phase.bmp
│   ├── edge_detection_phase.bmp
│   └── sharpen_phase.bmp
├── SLM_phase_images/             # Phase maps for various convolution kernels
│   ├── slm_blur_phase.bmp
│   ├── slm_edge_phase.bmp
│   ├── slm_gaussian_phase.bmp
│   ├── slm_laplacian_phase.bmp
│   └── slm_sharpen_phase.bmp
└── LICENSE                       # MIT License
```

## Installation Instructions

### System Requirements
- Python 3.7+
- PyTorch 1.7+
- NumPy
- OpenCV (cv2)
- Matplotlib
- Pillow (PIL)

### Installation Steps

1. Clone the project locally:
```bash
git clone https://gitee.com/mrbear_x/slmoperation.git
cd slmoperation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install torch numpy opencv-python matplotlib pillow
```

## Usage Guide

### 1. Basic Usage Example

```python
from SLM_ImageProcessor import SLM_ImageProcessor
import cv2

# Initialize SLM processor
slm = SLM_ImageProcessor(
    slm_resolution=(1920, 1080),
    wavelength=532e-9,
    pixel_size=8e-6
)

# Generate sharpening phase map
sharpen_phase = slm.generate_phase_for_image('sharpen')

# Visualize phase map
slm.visualize_phase_map(sharpen_phase, "Sharpening Phase")

# Save phase map
slm.save_phase_as_image(sharpen_phase, 'sharpen_phase.bmp')
```

### 2. Create Convolution Phase with Blazed Grating

```python
from create_slm_phase_for_convolution import create_slm_phase_for_convolution

# Generate sharpening convolution phase with blazed grating
combined_phase = create_slm_phase_for_convolution(
    kernel_name='sharpen',
    add_grating=True
)
```

### 3. Optical Propagation Simulation

```python
from OpticalPropagator import OpticalPropagator
import torch

# Initialize optical propagator
propagator = OpticalPropagator(
    wavelength=532e-9,
    pixel_size=8e-6,
    distance=0.1
)

# Simulate optical propagation
output_field = propagator.forward(phase_slm)
```

### 4. Manual Convolution Operation

```python
from main import manual_convolution
import cv2

# Load image
image = cv2.imread('images/bus.jpg', cv2.IMREAD_GRAYSCALE)

# Define convolution kernel
kernel = [[0, -1, 0],
          [-1, 5, -1],
          [0, -1, 0]]

# Perform convolution
result = manual_convolution(image, kernel)
```

### 5. Train Neural Network Phase Optimizer

```python
from OpticalPropagator import train_neural_phase_optimizer

# Train neural network
train_neural_phase_optimizer()
```

## Main Classes Overview

### SLM_ImageProcessor

The SLM_ImageProcessor is the core class of the project, providing the following methods:

| Method | Description |
|--------|-------------|
| `generate_kernel_phase(kernel_type)` | Generate phase map for specified convolution kernel type |
| `generate_phase_for_image(image, operation)` | Generate phase map for image processing operation |
| `generate_blazed_grating(period, direction)` | Generate blazed grating phase map |
| `generate_fresnel_lens(focal_length)` | Generate Fresnel lens phase map |
| `visualize_phase_map(phase_map, title)` | Visualize phase map |
| `save_phase_as_image(phase_map, filename)` | Save phase map as BMP image |

Supported kernel types:
- `sharpen`: Image sharpening
- `edge`: Edge detection
- `blur`: Blurring
- `laplacian`: Laplacian operator
- `gaussian`: Gaussian blur

### OpticalPropagator

The OpticalPropagator class simulates light wave propagation:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| wavelength | float | 532e-9 | Laser wavelength (meters) |
| pixel_size | float | 8e-6 | SLM pixel size (meters) |
| distance | float | 0.1 | Propagation distance (meters) |

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Project author: mrbear_x

## Contributions

Issues and Pull Requests are welcome to help improve this project.