import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Animator:
    def __init__(self, trajectory_1, ball_thrower, pixel_to_meter=0.01):
        self.trajectory_1 = trajectory_1  
        self.ball_thrower = ball_thrower  
        self.trajectory_2 = ball_thrower.throw_trajectory 
        self.target_point = ball_thrower.target_point 
        self.pixel_to_meter = pixel_to_meter
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ball1, = self.ax.plot([], [], 'ro', label="Ball 1 (Observed)")
        self.ball2, = self.ax.plot([], [], 'bo', label="Ball 2 (Throw)")
        self.target = None
        self.time_template = 'Time: {:.2f} s'
        self.time_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes)

    def init_animation(self):
        traj1_x = [point[1] for point in self.trajectory_1]
        traj1_y = [point[2] for point in self.trajectory_1]
        traj2_x = [point[1] for point in self.trajectory_2]
        traj2_y = [point[2] for point in self.trajectory_2]

        self.ax.set_xlim(0, max(traj1_x + traj2_x) + 1)
        self.ax.set_ylim(0, max(traj1_y + traj2_y) + 1)
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.set_title("Ball Motion Animation")
        self.ax.grid(True, linestyle="--", alpha=0.7)

        self.target = self.ax.scatter(
            self.target_point[1], self.target_point[2],
            color='green', label="Target Point", s=100
        )
        self.ax.plot(traj1_x, traj1_y, 'r--', alpha=0.5, label="Full Trajectory 1")
        self.ax.plot(traj2_x, traj2_y, 'b--', alpha=0.5, label="Full Trajectory 2")

        return self.ball1, self.ball2, self.target, self.time_text

    def update(self, frame):
        if frame < len(self.trajectory_1):
            self.ball1.set_data(
                [self.trajectory_1[frame][1]],
                [self.trajectory_1[frame][2]]
            )

        if frame < len(self.trajectory_2):
            self.ball2.set_data(
                [self.trajectory_2[frame][1]],
                [self.trajectory_2[frame][2]]
            )

        self.time_text.set_text(self.time_template.format(frame * 0.01))
        return self.ball1, self.ball2, self.target, self.time_text

    def animate(self):
        total_frames = max(len(self.trajectory_1), len(self.trajectory_2))
        anim = FuncAnimation(
            self.fig, self.update, frames=total_frames, init_func=self.init_animation, blit=True, interval=20
        )
        plt.show()



# Main function for the process, takes about maximum 20 seconds
# Functionality does not include saving mp4 files, output files are screen recorded, 
# but the animator class includes displaying canvas of the animation while running the application
if __name__ == "__main__":
    from ball_thrower import BallThrower
    from ball_detector import BallDetector

    video_path = "test_3.mp4"

    ball_detector = BallDetector(video_path, pixel_to_meter=0.01)

    if not ball_detector.trajectory:
        print("No trajectory detected.")
        exit()

    estimated_trajectory_1 = ball_detector.estimated_trajectory
    ball_thrower = BallThrower(ball_detector)

    animator = Animator(estimated_trajectory_1, ball_thrower)
    animator.animate()
