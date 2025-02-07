import cv2
import numpy as np
from gaussian import custom_gaussian_blur
from sobel import manual_sobel
import time

def fit_circle_manual(points, max_iter=100, tol=1e-6):
    def compute_residuals(params, points):
        cx, cy, r = params
        x, y = points[:, 0], points[:, 1]
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        return distances - r

    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    dists = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    r = np.mean(dists)

    params = np.array([cx, cy, r])

    for _ in range(max_iter):
        residuals = compute_residuals(params, points)
        
        J = np.zeros((len(points), 3))
        for i, (x, y) in enumerate(points):
            d = np.sqrt((x - params[0])**2 + (y - params[1])**2)
            if d < 1e-6:
                d = 1e-6
            J[i] = [-(x - params[0])/d, -(y - params[1])/d, -1]

        JTJ = J.T @ J
        lambda_reg = 1e-3
        try:
            delta = np.linalg.solve(JTJ + lambda_reg * np.eye(3), -J.T @ residuals)
            if np.linalg.norm(delta) < tol:
                break
            params += delta
        except np.linalg.LinAlgError:
            break

    return params

def create_canvas_with_circles(image_path, circles):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    dimmed_image = (image * 0.3).astype(np.uint8)
    meters_per_pixel = 0.01 
    canvas = dimmed_image.copy()
    for x in range(0, width, int(1 / meters_per_pixel)):
        cv2.line(canvas, (x, 0), (x, height), (200, 200, 200), 1)  
        if x % int(10 / meters_per_pixel) == 0: 
            cv2.putText(canvas, f"{x * meters_per_pixel:.0f}m", (x, height - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    for y in range(0, height, int(1 / meters_per_pixel)):
        cv2.line(canvas, (0, y), (width, y), (200, 200, 200), 1)  
        if y % int(10 / meters_per_pixel) == 0:  
            cv2.putText(canvas, f"{y * meters_per_pixel:.0f}m", (10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for x, y, radius in circles:  
        cv2.circle(canvas, (int(x), int(y)), int(radius), (0, 255, 0), 2) 
    return canvas

def detect_balls_and_display_canvas(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = custom_gaussian_blur(gray, (5, 5), 1.5)
    binary_edges = manual_sobel(blurred)
    binary_edges = (binary_edges > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output_image = image.copy()
    min_points = 30
    min_radius = 10
    max_radius = 100
    detected_circles = []
    print(f"Found {len(contours)} contours")
    
    for i, contour in enumerate(contours):
        if len(contour) < min_points:
            continue
        points = contour.reshape(-1, 2)
        try:
            cx, cy, radius = fit_circle_manual(points)
            print(f"Contour {i}: points={len(points)}, radius={radius:.1f}")
            
            if min_radius <= radius <= max_radius:
                detected_circles.append((int(cx), int(cy), int(radius)))  # Store as (x, y, radius)
        except Exception as e:
            print(f"Error on contour {i}: {e}")
    print(f"Detected {len(detected_circles)} circles")

    detected_circles_array = [[x, y, radius] for x, y, radius in detected_circles]
    print("Detected Circles Array:", detected_circles_array)
    canvas = create_canvas_with_circles(image_path, detected_circles)
    return canvas, detected_circles_array

if __name__ == "__main__":
    image_path = "input_image3.png"
    canvas, circles_data = detect_balls_and_display_canvas(image_path)
    cv2.imshow("Canvas", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()