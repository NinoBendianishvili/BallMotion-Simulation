
# Hit a Ball to The Fixed Target Project

## Overview
This project combines image processing, numerical simulation, and animation to model, detect, and visualize the motion of balls in a 2D environment. The workflow includes detecting circular objects in an image, calculating their trajectories for a newly created ball to hit the target balls using physical principles, and creating an animation to visualize their motion and interactions. The implementation relies on Python with key libraries like OpenCV and NumPy for computational efficiency and visualization.

## 1. Circle Detection

### **Objective**
To identify circular objects in a given input image and determine their locations and radii. This is a prerequisite for simulating their motion.

### **Key Methods**

#### **Image Preprocessing**
- The input image is read and converted to grayscale to simplify processing. This step reduces computational complexity and improves edge detection.

#### **Gaussian Blurring**
- Noise in the image is reduced by applying a Gaussian blur. The custom Gaussian blur function convolves this kernel with the image.

```math
B(i,j) = \sum_{m=-k_h}^{k_h} \sum_{n=-k_w}^{k_w} I(i+m, j+n) \cdot \hat{G}(m,n)
```

Where:
- \( B(i,j) \): The resulting blurred intensity at pixel \((i,j)\).
- \( I(i+m,j+n) \): The intensity of the pixel in the neighborhood of \((i,j)\), with \(m, n\) as offsets.
- \( G(m,n) \): The normalized Gaussian kernel value at offset \((m,n)\).
- \( k_h, k_w \): The half-dimensions of the kernel (height and width).

#### **Edge Detection**
- A custom Sobel operator is applied to compute the image gradient in the x and y directions. The gradient magnitude is used to create a binary edge map.

```math
M(i, j) = \sqrt{G_x(i, j)^2 + G_y(i, j)^2}
```

A Binary edge map is created using the threshold:

```math
E(i, j) = \begin{cases}
255, & \text{if } M_{normalized}(i, j) > T \\
0, & \text{otherwise}
\end{cases}
```

where \( M_{normalized} \) is:

```math
M_{normalized}(i, j) = \frac{M(i, j) - M_{min}}{M_{max} - M_{min}} \cdot (new_{max} - new_{min}) + new_{min}
```

#### **Contour Detection**
- Contours are identified in the binary image, and points belonging to each contour are extracted.

#### **Circle Fitting**
- A least-squares optimization algorithm is used to fit a circle to the contour points. The circle parameters \((c_x, c_y, r)\) are iteratively refined to minimize the residual error.

### **Outputs**
- **Detected Circles:** Each circle is represented as \((c_x, c_y, r)\), where \(c_x, c_y\) are the circle's center coordinates, and \(r\) is its radius.
- **Canvas Visualization:** A dimmed version of the original image with detected circles overlaid and a grid for reference.

## 2. Trajectory Calculation

### **Objective**
To compute the trajectory of a ball considering gravity and air resistance, enabling simulation of its motion from an initial position to a target.

### **Mathematical Models**

#### **Ball Motion ODE:**
The ball's position \((x, y)\) and velocity \((v_x, v_y)\) evolve over time according to:

```math
\frac{dv_x}{dt} = - \left(\frac{k}{m}\right) v_x \sqrt{v_x^2 + v_y^2}
\frac{dv_y}{dt} = - g - \left(\frac{k}{m}\right) v_y \sqrt{v_x^2 + v_y^2}
```

#### **Numerical Integration: Runge-Kutta 4th Order (RK4)**
For each time step \(h = 0.01\):

```math
y_1 = y_0 + \frac{1}{6} (k_1 + 2k_2 + 2k_3 + k_4)
```

where:
- \(k_1 = h f(x_0, y_0)\)
- \(k_2 = h f(x_0 + \frac{1}{2} h, y_0 + \frac{1}{2} k_1)\)
- \(k_3 = h f(x_0 + \frac{1}{2} h, y_0 + \frac{1}{2} k_2)\)
- \(k_4 = h f(x_0 + h, y_0 + k_3)\)

#### **Shooting Method**
- An iterative approach computes the initial velocity \(v_0\) that results in the ball reaching a target \((x_t, y_t)\) within time \(T\).
- Errors in position are minimized by solving:

```math
E_x = X_{final} - X_{target}; \quad E_y = Y_{final} - Y_{target}
```

The Jacobian matrix approximates the relationship between velocity changes and errors, enabling corrections:

```math
J =
\begin{bmatrix}
\frac{e_x(v + [\epsilon, 0]) - e_x(v)}{\epsilon} & \frac{e_x(v + [0, \epsilon]) - e_x(v)}{\epsilon} \\
\frac{e_y(v + [\epsilon, 0]) - e_y(v)}{\epsilon} & \frac{e_y(v + [0, \epsilon]) - e_y(v)}{\epsilon}
\end{bmatrix}
```

### **Outputs**
- **Trajectories:** A sequence of \((x, y, v_x, v_y)\) states for each target ball.

## 3. Animation

### **Objective**
To create a visual representation of the simulated trajectories and interactions between the balls.

### **Process**
1. **Setup:** The detected circles are drawn on the initial canvas. A red ball is introduced at the bottom center.
2. **Frame-by-Frame Update:** The red ball's position is updated based on the computed trajectory.
3. **Collision Detection:** If the red ball's distance from a target ball's center is less than the sum of their radii, a collision is detected, and the target ball is removed.
4. **Video Generation:** Frames are saved to create an animation illustrating the motion.

### **Outputs**
- **Animation:** A smooth animation showing the red ball’s motion and its collisions with other balls. The output is stored in `Task1/outputs`.

To run the program for personal tests, execute the `animation` class and modify the `image_path` variable.

## 4. Numerical Experiments

### **Comparison of Numerical Methods**

As an additional numerical experiment, the code provides implementations of RK4, RK2, and Euler methods for solving the Ball Motion ODE. Results:

| Method  | Max Error | Mean Error | Computation Time (s) |
|---------|----------|-----------|------------------|
| Euler   | 0.242665 | 0.242665  | 0.002268        |
| RK2     | 0.000204 | 0.000204  | 0.004848        |
| RK4     | 0.000000 | 0.000000  | 0.011410        |

### **Conclusion:**
- Euler method is the fastest but has a large error, making it unsuitable for high-accuracy applications.
- RK2 reduces the error but remains less accurate than RK4.
- RK4 is the most precise and ideal for this simulation, despite taking slightly longer to compute.


