import numpy as np

def create_gaussian_kernel(kernel_size, sigma):
    if isinstance(kernel_size, tuple):
        kernel_height, kernel_width = kernel_size
    else:
        kernel_height = kernel_width = kernel_size
    if kernel_height % 2 == 0:
        kernel_height += 1
    if kernel_width % 2 == 0:
        kernel_width += 1
    kh = (kernel_height - 1) // 2
    kw = (kernel_width - 1) // 2
    x, y = np.meshgrid(
        np.linspace(-kw, kw, kernel_width),
        np.linspace(-kh, kh, kernel_height)
    )
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gaussian / gaussian.sum()

def custom_gaussian_blur(image, kernel_size=(5, 5), sigma=1.5):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    height, width = image.shape
    if isinstance(kernel_size, tuple):
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
    else:
        pad_h = pad_w = kernel_size // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    blurred = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            roi = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            blurred[i, j] = np.sum(roi * kernel)
    
    return blurred.astype(np.uint8)