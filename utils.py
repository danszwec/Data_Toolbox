import cv2
import numpy as np
import os
from datetime import datetime



def compute_avg_motion(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                        pyr_scale=0.5, levels=4, winsize=25,
                                        iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)


def overlay_text_on_frame(frame, text, position=(10, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green
    thickness = 2
    return cv2.putText(frame.copy(), text, position, font, font_scale, color, thickness, cv2.LINE_AA)


def optical_flow_statistics(moving_motion_list: list ,still_motion_list: list,save_dir: str) -> None:

    move_count = 0
    still_count = 0
    if os.path.exists(os.path.join(save_dir, "move.txt")):
        move_count = len(open(os.path.join(save_dir, "move.txt")).readlines())
    if os.path.exists(os.path.join(save_dir, "still.txt")):
        still_count = len(open(os.path.join(save_dir, "still.txt")).readlines())

    with open(os.path.join(save_dir, "optical_flow_statistics.txt"), "w") as file:
        #write title
        date = datetime.now().strftime('%Y-%m-%d')
        file.write(f"Optical Flow Statistics\n date: {date}\n")
        file.write("=====================\n")
        
        #write the number of rows on save dir still.txt and bad.txt
        file.write(f"Still Count: {still_count}\n {still_count/len(still_motion_list)*100}%")
        file.write(f"move Count: {move_count}\n {move_count/len(moving_motion_list)*100}%")
        

        #write the moving avg motion and still avg motion
        if len(moving_motion_list) != 0:
            file.write(f"Moving Avg Motion: {np.mean(moving_motion_list):.4f}\n")
            file.write(f"Moving Std Motion: {np.std(moving_motion_list):.4f}\n")
        else:
            file.write("Moving Avg Motion: 0.0000\n")
            file.write("Moving Std Motion: 0.0000\n")

        if len(still_motion_list) != 0:
            file.write(f"Still Avg Motion: {np.mean(still_motion_list):.4f}\n")
            file.write(f"Still Std Motion: {np.std(still_motion_list):.4f}\n")
        else:
            file.write("Still Avg Motion: 0.0000\n")
            file.write("Still Std Motion: 0.0000\n")
