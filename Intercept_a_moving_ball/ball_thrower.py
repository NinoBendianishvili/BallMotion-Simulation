import numpy as np
import matplotlib.pyplot as plt
import random
from ball_detector import BallDetector
import cv2

class BallThrower:
    def __init__(self, ball_detector, target_fraction=0.5, g=9.81):
        self.pixel_to_meter = ball_detector.pixel_to_meter
        self.target_fraction = target_fraction
        self.g = g

        self.target_point = self._select_target_point(ball_detector.estimated_trajectory)
        print(f"\ntarget point is: {self.target_point}")
        self.t_target = self.target_point[0]

        self.initial_vx, self.initial_vy = self._calculate_velocity(
            self.target_point[1:], (0, 0), self.t_target, self.g
        )

        self.initial_state = [0, 0, self.initial_vx, self.initial_vy]
        self.throw_trajectory = self._estimate_trajectory(self.initial_state, (0, self.t_target), 0.01, 0.01, self.g)

    def _select_target_point(self, estimated_trajectory):
        second_half = estimated_trajectory[len(estimated_trajectory) // 2:]
        return second_half[int(len(second_half) * self.target_fraction)]

    def _calculate_velocity(self, target_point, start_point, t_target, g):
        x_target, y_target = target_point
        x_start, y_start = start_point
        dt = 0.01
        initial_vx, initial_vy = 10.0, 10.0  # Initial guesses

        for _ in range(100):  
            trajectory = self._estimate_trajectory(
                [x_start, y_start, initial_vx, initial_vy], (0, t_target), dt, 0.01, g
            )
            x_end, y_end = trajectory[-1][1], trajectory[-1][2] 
            error_x = x_target - x_end
            error_y = y_target - y_end
            if abs(error_x) < 0.01 and abs(error_y) < 0.01:
                break

            initial_vx += 0.1 * error_x / t_target
            initial_vy += 0.1 * error_y / t_target
        print(f"\nInitial velocities of thrown ball: vx={initial_vx} vy={initial_vy}")
        return initial_vx, initial_vy

    def _trajectory_ode(self, t, state, km, g):
        x, y, vx, vy = state
        speed = np.sqrt(vx**2 + vy**2)
        dx_dt = vx
        dy_dt = vy
        dvx_dt = -km * vx * speed
        dvy_dt = -g - km * vy * speed
        return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])

    def _rk4_step(self, f, t, y, h, km, g):
        k1 = f(t, y, km, g)
        k2 = f(t + h / 2, y + h * k1 / 2, km, g)
        k3 = f(t + h / 2, y + h * k2 / 2, km, g)
        k4 = f(t + h, y + h * k3, km, g)
        return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _estimate_trajectory(self, initial_state, t_span, dt, km, g):
        t_values = [t_span[0]]
        y_values = [initial_state]
        t = t_span[0]
        state = initial_state

        while t < t_span[1]:
            state = self._rk4_step(self._trajectory_ode, t, state, dt, km, g)
            t += dt
            t_values.append(t)
            y_values.append(state)
            if state[1] < 0: 
                break

        trajectory = [(t_values[i], y_values[i][0], y_values[i][1]) for i in range(len(t_values))]
        return trajectory

    def plot_trajectories(self, estimated_trajectory):
        plt.figure(figsize=(10, 6))
        est_x = [point[1] for point in estimated_trajectory]
        est_y = [point[2] for point in estimated_trajectory]
        plt.plot(est_x, est_y, label="Estimated Trajectory", color="red", linestyle="--")
        throw_x = [point[1] for point in self.throw_trajectory]
        throw_y = [point[2] for point in self.throw_trajectory]
        plt.plot(throw_x, throw_y, label="Throw Trajectory", color="blue")
        plt.scatter(self.target_point[1], self.target_point[2], color="green", label="Target Point", s=100, zorder=5)
        plt.title("Trajectory Comparison", fontsize=16)
        plt.xlabel("X Position (m)", fontsize=12)
        plt.ylabel("Y Position (m)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


# # Code for testing
# ball_detector = BallDetector("test_2.mp4", pixel_to_meter=0.01)
# if not ball_detector.trajectory:
#     print("No trajectory detected.")
#     exit()

# ball_thrower = BallThrower(ball_detector)
# print(f"Target Point: {ball_thrower.target_point}")
# print(f"Initial Velocity: vx = {ball_thrower.initial_vx:.2f}, vy = {ball_thrower.initial_vy:.2f}")
# ball_thrower.plot_trajectories(ball_detector.estimated_trajectory)
