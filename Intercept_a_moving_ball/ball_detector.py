import cv2
import numpy as np
import matplotlib.pyplot as plt

class BallDetector:
    def __init__(self, video_path, pixel_to_meter=1/100):
        self.video_path = video_path
        self.pixel_to_meter = pixel_to_meter

        self.prev_frame = None
        self.start_time = None
        self.trajectory = []
        self.estimated_trajectory = None
        self.vx = None
        self.vy = None
        self.km = None
        self.radius = None

        self._process_video()

    def _process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        dt = 0.01
        g = 9.81

        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            largest_contour, thresh, trajectory = self.detect_ball(frame, frame_time)

            if largest_contour is not None:
                self.radius = self._calculate_radius(largest_contour)

        cap.release()
        self.vx, self.vy = self.gradient_descent_initial_velocity()
        self.km = self.shooting_method(self.vx, self.vy, dt, g)
        self.estimated_trajectory = self.estimate_trajectory(self.vx, self.vy, dt, self.km, g)

    def detect_ball(self, frame, frame_time):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            self.start_time = frame_time
            return None, None, None

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 3, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_frame = gray

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 50:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    t = frame_time - self.start_time
                    adjusted_cy = (frame.shape[0] - cy) * self.pixel_to_meter
                    self.trajectory.append((t, cx * self.pixel_to_meter, adjusted_cy))
                return largest_contour, thresh, self.trajectory

        return None, thresh, self.trajectory

    def _calculate_radius(self, contour):
        _, radius = cv2.minEnclosingCircle(contour)
        return radius * self.pixel_to_meter

    def gradient_descent_initial_velocity(self, learning_rate=0.01, max_iterations=1000, g=9.81):
        if len(self.trajectory) < 2:
            return 0, 0

        def trajectory_error(params):
            vx, vy = params
            estimated = self.estimate_trajectory(vx, vy, dt=0.01, km=0.01, g=g)

            observed = np.array(self.trajectory)
            if len(estimated) < len(observed):
                return float('inf')

            observed_x = [point[1] for point in observed]
            observed_y = [point[2] for point in observed]
            estimated_x = [point[1] for point in estimated[:len(observed)]]
            estimated_y = [point[2] for point in estimated[:len(observed)]]

            return np.sum((np.array(observed_x) - np.array(estimated_x))**2 + \
                          (np.array(observed_y) - np.array(estimated_y))**2)

        vx, vy = 0, 0  # Initial guesses
        for _ in range(max_iterations):
            error = trajectory_error((vx, vy))

            # Partial derivatives, numerical gradient
            d_error_dvx = (trajectory_error((vx + 1e-5, vy)) - error) / 1e-5
            d_error_dvy = (trajectory_error((vx, vy + 1e-5)) - error) / 1e-5
            vx -= learning_rate * d_error_dvx
            vy -= learning_rate * d_error_dvy

            if abs(d_error_dvx) < 1e-5 and abs(d_error_dvy) < 1e-5:
                break
        print(f"\nInitial velocities of the detected ball: vx={vx} vy={vy}")
        return vx, vy

    def shooting_method(self, initial_vx, initial_vy, dt, g):
        km = 0.1  # Initial guess for k/m
        error_threshold = 0.001
        learning_rate = 0.05

        def calculate_last_point_error(observed, estimated):
            last_observed = observed[-1]
            last_observed_x = last_observed[1]
            closest_point = min(estimated, key=lambda point: abs(point[1] - last_observed_x))
            error = last_observed[2] - closest_point[2]
            return error

        for iteration in range(200): 
            estimated_trajectory = self.estimate_trajectory(initial_vx, initial_vy, dt, km, g)

            observed = np.array(self.trajectory)
            if len(observed) > 1 and len(estimated_trajectory) > 1:
                error = calculate_last_point_error(observed, estimated_trajectory)
                if abs(error) < error_threshold:
                    break

                km -= learning_rate * error
        print(f"\nEstimated k/m={km}")
        return km

    def ode_solver(self, f, t_span, y0, h):
        t_values = [t_span[0]]
        y_values = [y0]
        t = t_span[0]
        y = y0
        while t < t_span[1]:
            y = self.rk4_step(f, t, y, h)
            t += h
            t_values.append(t)
            y_values.append(y)
        return np.array(t_values), np.array(y_values)

    def rk4_step(self, f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def trajectory_ode(self, t, state, km, g):
        x, y, vx, vy = state
        speed = np.sqrt(vx**2 + vy**2)
        dx_dt = vx
        dy_dt = vy
        dvx_dt = -km * vx * speed
        dvy_dt = -g - km * vy * speed
        return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])

    def estimate_trajectory(self, vx, vy, dt, km, g):
        x, y = self.trajectory[0][1], self.trajectory[0][2]
        initial_state = np.array([x, y, vx, vy])
        t_span = (0, 5)

        def f(t, state):
            return self.trajectory_ode(t, state, km, g)

        t_values, y_values = self.ode_solver(f, t_span, initial_state, dt)
        estimated_trajectory = [(t, state[0], state[1]) for t, state in zip(t_values, y_values) if state[1] >= 0]

        return estimated_trajectory

    def display_trajectory(self, video_width, video_height):
        if not self.trajectory:
            print("No trajectory to display.")
            return

        times, xs, ys = zip(*self.trajectory)

        plt.figure(figsize=(8, 8 * video_height / video_width))
        plt.plot(xs, ys, marker='o', label="Observed Trajectory")

        if self.estimated_trajectory:
            est_times, est_xs, est_ys = zip(*self.estimated_trajectory)
            plt.plot(est_xs, est_ys, marker='x', linestyle='--', label="Estimated Trajectory")

        plt.xlim(0, 6)
        plt.ylim(0, 6)
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Ball Trajectory")
        plt.legend()
        plt.grid()
        plt.gca().set_aspect(aspect='equal')
        plt.show()

# # Usage example
# ball_detector = BallDetector("test_2.mp4", pixel_to_meter=0.01)
# print(f"Initial Velocity: vx = {ball_detector.vx:.2f} m/s, vy = {ball_detector.vy:.2f} m/s")
# print(f"Drag Coefficient (k/m): {ball_detector.km:.4f}")
# print(f"Ball Radius: {ball_detector.radius:.4f} m")
# ball_detector.display_trajectory(video_width=1920, video_height=1080)
