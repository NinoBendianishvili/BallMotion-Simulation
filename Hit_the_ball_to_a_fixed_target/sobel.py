import numpy as np

def manual_sobel(image, thresh_value=50, new_min=0, new_max=255):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows, cols = image.shape
    grad_x = np.zeros_like(image, dtype=np.float64)
    grad_y = np.zeros_like(image, dtype=np.float64)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i - 1:i + 2, j - 1:j + 2] 
            grad_x[i, j] = np.sum(region * sobel_x)
            grad_y[i, j] = np.sum(region * sobel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    old_min, old_max = magnitude.min(), magnitude.max()
    normalized = (magnitude - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    normalized = normalized.astype(np.uint8)
    binary_edges = np.zeros_like(normalized, dtype=np.uint8)
    binary_edges[normalized > thresh_value] = 255

    return binary_edges


