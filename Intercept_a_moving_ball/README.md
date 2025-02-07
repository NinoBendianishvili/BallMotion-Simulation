# Intercept a Moving Ball Project Documentation

## Overview

This project implements a complete workflow for detecting, analyzing, and animating the motion of balls in a 2D environment. The system has three main components:

1. **Ball Detection** using `BallDetector` to detect a thrown ball and reconstruct its trajectory.
2. **Ball Throw Simulation** using `BallThrower` to simulate a throw aimed at hitting the previously detected ball.
3. **Visualization & Animation** using `Animator` to display the observed and simulated trajectories.

---

## 1. Ball Detection (`BallDetector`)

### Objective
Extracts the trajectory of a ball from a video by:
- Detecting the ball's position frame-by-frame.
- Estimating the ball's initial velocity and drag coefficient `(k/m)` using optimization methods.

### Key Methods

- **`_process_video()`**: Processes each video frame, detects the ball using contour analysis, and records its trajectory in meters (1 meter = 100 pixels).
- **`detect_ball()`**: Uses frame differencing and thresholding with OpenCV (`cv2`) functions like Gaussian blur, thresholding, and `findContours`.
- **`gradient_descent_initial_velocity()`**: Computes the ball's initial velocity `(v_x, v_y)` using gradient descent:
  
  $$ v_x(k+1) = v_x(k) - \eta \frac{\partial E}{\partial v_x} $$
  $$ v_y(k+1) = v_y(k) - \eta \frac{\partial E}{\partial v_y} $$
  
  - Initial velocity: `(0,0)`
  - Step size: `dt = 0.01`
  - Learning rate: `0.01`
  - Maximum iterations: `1000`
  - Error function:
    
    $$ E = \sum ((x_{obs} - x_{sim})^2 + (y_{obs} - y_{sim})^2) $$
  
- **`shooting_method()`**: Refines the drag coefficient `(k/m)` by aligning the final simulated trajectory point with the observed one.
  - Initial guess: `0.1`
  - Step size: `0.05`
  - Error threshold: `0.001`
- **`estimate_trajectory()`**: Simulates the ball's trajectory using numerical integration (RK4) to solve the motion equations.

### Mathematical Models

#### Ball Motion ODE

$$ \frac{dv_x}{dt} = - \left(\frac{k}{m} \right) v_x \sqrt{v_x^2 + v_y^2} $$
$$ \frac{dv_y}{dt} = -g - \left(\frac{k}{m} \right) v_y \sqrt{v_x^2 + v_y^2} $$

#### Runge-Kutta 4th Order (RK4) Integration

$$ y_1 = y_0 + \frac{1}{6} (k_1 + 2k_2 + 2k_3 + k_4) $$

where:

$$ k_1 = hf(x_0, y_0) $$
$$ k_2 = hf(x_0 + \frac{1}{2}h, y_0 + \frac{1}{2}k_1) $$
$$ k_3 = hf(x_0 + \frac{1}{2}h, y_0 + \frac{1}{2}k_2) $$
$$ k_4 = hf(x_0 + h, y_0 + k_3) $$

The Runge-Kutta 4th-order method (RK4) used in the code is not A-stable. It is conditionally stable, meaning that its stability depends on the step size h. If h is too large for a given problem, particularly for stiff equations, RK4 may become unstable. But as h=0.01, relatively small step size for this problem, RK4 can guarantee stability of the process. 

#### Outputs
- **Observed Trajectory**: A sequence of `(t, x, y)` points.
- **Estimated Trajectory**: A refined sequence matching the observed motion.
### Parameters:

- **Example 1**  
  - Estimated \( k/m \) = 0.18300996477505654  
  - Initial Velocity: \( v_x = -3.41 \) m/s, \( v_y = 5.07 \) m/s  
  - Drag Coefficient (\( k/m \)): 0.1830  
  - Ball Radius: 0.2831 m

![image](https://github.com/user-attachments/assets/09af85c2-a046-4c70-8235-af0cd323f1e4)

- **Example 2**  
  - Estimated \( k/m \) = 0.027051696461635643  
  - Initial Velocity: \( v_x = 5.23 \) m/s, \( v_y = 0.90 \) m/s  
  - Drag Coefficient (\( k/m \)): 0.0271  
  - Ball Radius: 0.3289 m  

![image](https://github.com/user-attachments/assets/82fddcbb-0168-4145-a1b4-b1695d508b1b)

---

## 2. Ball Throw Simulation (`BallThrower`)

### Objective
Simulates a ball throw to hit a specific target point extracted from the observed trajectory.

### Key Methods

- **`_select_target_point()`**: Randomly selects a target from the second half of the observed trajectory for better visualization.
- **`_calculate_velocity()`**: Determines initial velocity `(v_x, v_y)` needed to reach the target using iterative refinement:
  
  - `dt = 0.01`
  - Assumed `k/m = 0.01`
  - Error calculation:

$$
E_x = x_{target} - x_{end}  
E_y = y_{target} - y_{end}
$$

  
  - Velocity update (100 iterations, error threshold `0.01`):
    
$$
v_x += 0.1 \frac{E_x}{t_{target}}  
v_y += 0.1 \frac{E_y}{t_{target}}
$$
  

- **`_estimate_trajectory()`**: Computes the simulated trajectory using RK4 integration.

## Outputs

1. **Simulated Throw Trajectory**: A sequence of \((t, x, y)\) points for the ball’s motion toward the target point.

2. **Initial Velocity**: Components \( v_x \) and \( v_y \) of the thrown ball.

3. **Target Point**: The chosen goal coordinates \((x, y)\) to be hit by the ball.

### Example 1:
- **Target Point**: (1.0800000000000007, 3.104110757167739, 2.0603780941512024)
- **Initial Velocity**: \( v_x = 2.96 \), \( v_y = 7.33 \)

![image](https://github.com/user-attachments/assets/2fbf1003-e064-424b-85d8-48ba13220963)

### Example 2:
- **Target Point**: (0.8600000000000005, 4.276544362129806, 2.2943214058925854)
- **Initial Velocity**: \( v_x = 5.13 \), \( v_y = 7.02 \)

![image](https://github.com/user-attachments/assets/7321b449-afbb-4b69-a412-54f54a0efaad)

---

## 3. Animation (`Animator`)

### Functionality
Visualizes the observed and simulated trajectories and compares them in an animated plot.

### Key Methods

- **`init_animation()`**: Sets axis limits, gridlines, and initializes the animation.
- **`update(frame)`**: Updates the positions of the observed and thrown balls per frame.
- **`animate()`**: Generates a frame-by-frame animation of the observed and simulated motions.

### Outputs
- **`Animation`**: A dynamic visualization showing the trajectories of the observed and simulated balls and their alignment with the target point. Animations are stored in Task2/outputs folder. To run code for personal tests one should run animator class and change value of video_path variable

---

## Summary
This project effectively detects, simulates, and visualizes ball trajectories using numerical methods like gradient descent, RK4 integration, and optimization techniques. It provides insights into trajectory estimation, drag coefficient calculation, and impact prediction.
