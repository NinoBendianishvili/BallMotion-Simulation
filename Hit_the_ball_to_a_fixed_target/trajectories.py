from circle_detection import detect_balls_and_display_canvas
import numpy as np
import cv2

mass = 1  
air_resistance = 0.01  
g = 9.81  
PIXELS_PER_METER = 100 

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def solve_ode(f, t_span, y0, h):
    t_values = [t_span[0]]
    y_values = [y0]
    t = t_span[0]
    y = y0
    while t < t_span[1]:
        y = rk4_step(f, t, y, h)
        t += h
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

def ball_trajectory(t, state, k, m, g):
    x, y, vx, vy = state
    speed = np.sqrt(vx**2 + vy**2)
    dx_dt = vx
    dy_dt = vy
    dvx_dt = -(k / m) * vx * speed
    dvy_dt = - g - (k / m) * vy * speed 
    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])

def shooting_method(x0, y0, xt, yt, T, k, m, g, v_guess, h, tol=1e-5, max_iter=100):
    def target_error(v0):
        state0 = [x0, y0, v0[0], v0[1]]
        _, trajectory = solve_ode(lambda t, state: ball_trajectory(t, state, k, m, g), (0, T), state0, h)
        final_x, final_y = trajectory[-1, :2]
        return np.array([final_x - xt, final_y - yt])

    v = np.array(v_guess, dtype=float)
    for _ in range(max_iter):
        error = target_error(v)
        if np.linalg.norm(error) < tol:
            return v
        epsilon = 1e-6
        jacobian = np.zeros((2, 2))
        for i in range(2):
            v_plus = v.copy()
            v_plus[i] += epsilon
            error_plus = target_error(v_plus)
            jacobian[:, i] = (error_plus - error) / epsilon
        
        try:
            delta_v = np.linalg.solve(jacobian, -error)
            v += delta_v
        except np.linalg.LinAlgError:
            v += -error * 0.1
            
    raise RuntimeError("Shooting method did not converge.")

def convert_to_physics_coordinates(point, canvas_height):
    return (point[0] / PIXELS_PER_METER, (canvas_height - point[1]) / PIXELS_PER_METER)

def convert_to_screen_coordinates(point, canvas_height):
    return (int(point[0] * PIXELS_PER_METER), int(canvas_height - point[1] * PIXELS_PER_METER))

def find_trajectories_for_all_balls(circles_data, canvas_dims, k, m, g, h, T):
    print("Finding trajectories for all balls...")
    trajectories_map = {}
    canvas_height, canvas_width = canvas_dims

    x0, y0 = convert_to_physics_coordinates((canvas_width // 2, canvas_height - 20), canvas_height)

    for target_ball in circles_data:
        cx, cy, radius = target_ball
        xt, yt = convert_to_physics_coordinates((cx, cy), canvas_height)
        print(f"Starting trajectory calculation for target ball at ({cx}, {cy})")
        dx = xt - x0
        dy = yt - y0
        T_guess = np.sqrt(2 * abs(dy) / g) * 1.5
        vx_guess = dx / T_guess
        vy_guess = dy / T_guess + g * T_guess / 2
        v_guess = [vx_guess, vy_guess]
        
        try:
            v0 = shooting_method(x0, y0, xt, yt, T_guess, k, m, g, v_guess, h)
            state0 = [x0, y0, v0[0], v0[1]]
            _, trajectory = solve_ode(lambda t, state: ball_trajectory(t, state, k, m, g), (0, T_guess), state0, h)
            print(f"Trajectory found for target ball at ({cx}, {cy})")
            trajectories_map[tuple(v0)] = trajectory
        except RuntimeError:
            print(f"No trajectory found for target ball at ({cx}, {cy})")

    print("All trajectories calculated.")
    return trajectories_map

def draw_trajectories_on_canvas(canvas, trajectories_map):
    canvas_height = canvas.shape[0]
    for v0, trajectory in trajectories_map.items():
        screen_points = [convert_to_screen_coordinates((x, y), canvas_height) 
                        for x, y, _, _ in trajectory]
        points = np.array(screen_points, dtype=np.int32)

        for i in range(len(points) - 1):
            cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), (255, 0, 0), 2)
        
        for i in range(0, len(points), 5):
            cv2.circle(canvas, tuple(points[i]), 3, (0, 255, 0), -1)


if __name__ == "__main__":
    image_path = "input_image3.png"
    canvas, circles_data = detect_balls_and_display_canvas(image_path)
    canvas_height, canvas_width, _ = canvas.shape
    initial_pos = (canvas_width // 2, canvas_height - 20)
    cv2.circle(canvas, initial_pos, 10, (0, 0, 255), -1)  

    k = air_resistance
    m = mass
    h = 0.01  # Time step 
    T = 2.0  # Maximum time 

    if circles_data:
        trajectories_map = find_trajectories_for_all_balls(circles_data, (canvas_height, canvas_width), k, m, g, h, T)
        draw_trajectories_on_canvas(canvas, trajectories_map)

    cv2.imshow("Canvas with Trajectories", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()