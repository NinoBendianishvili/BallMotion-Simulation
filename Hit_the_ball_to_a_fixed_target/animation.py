import cv2
import numpy as np
from circle_detection import detect_balls_and_display_canvas
from trajectories import (find_trajectories_for_all_balls, convert_to_screen_coordinates, mass, air_resistance, g, PIXELS_PER_METER)
import os

def animate_trajectory(image, trajectories_map, circles_data, output_video_path):
    canvas_height, canvas_width, _ = image.shape

    red_ball_radius = 10
    red_ball_pos = (canvas_width // 2, canvas_height - 20)
    footprint_canvas = image.copy()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (canvas_width, canvas_height))
    for v0, trajectory in trajectories_map.items():
        target_ball = None
        for circle in circles_data:
            cx, cy, _ = circle
            if np.isclose(v0[0], cx / PIXELS_PER_METER, atol=0.1) and np.isclose(v0[1], (canvas_height - cy) / PIXELS_PER_METER, atol=0.1):
                target_ball = circle
                break

        for t_idx, state in enumerate(trajectory):
            x, y = state[:2]
            red_ball_screen_pos = convert_to_screen_coordinates((x, y), canvas_height)
            canvas_copy = image.copy()
            for circle in circles_data:
                cx, cy, radius = circle
                cv2.circle(canvas_copy, (cx, cy), radius, (0, 0, 255), 2) 
            cv2.circle(footprint_canvas, red_ball_screen_pos, 2, (255, 0, 0), -1)
            combined_canvas = cv2.addWeighted(footprint_canvas, 0.5, canvas_copy, 0.5, 0)
            cv2.circle(combined_canvas, red_ball_screen_pos, red_ball_radius, (0, 0, 255), -1)
            if target_ball:
                cx, cy, radius = target_ball
                distance = np.sqrt((red_ball_screen_pos[0] - cx)**2 + (red_ball_screen_pos[1] - cy)**2)
                if distance <= radius + red_ball_radius:
                    circles_data.remove(target_ball)
                    break

            video_writer.write(combined_canvas)
            cv2.imshow("Animation", combined_canvas)
            key = cv2.waitKey(30)  
            if key == 27: 
                video_writer.release()
                cv2.destroyAllWindows()
                return
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "input_image3.png"  #Image path
    image, circles_data = detect_balls_and_display_canvas(image_path)
    canvas_height, canvas_width, _ = image.shape
    initial_pos = (canvas_width // 2, canvas_height - 20)
    cv2.circle(image, initial_pos, 10, (0, 0, 255), -1)  

    k = air_resistance
    m = mass
    h = 0.01  # Time step 
    T = 2.0  # Maximum time 
    output_folder = "./Task1/outputs"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_video_path = os.path.join(output_folder, "output_video3.mp4")
    if circles_data:
        trajectories_map = find_trajectories_for_all_balls(circles_data, (canvas_height, canvas_width), k, m, g, h, T)
        animate_trajectory(image, trajectories_map, circles_data, output_video_path)
