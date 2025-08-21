import os
from pathlib import Path
from datetime import datetime
import cv2

import numpy as np
from tqdm import tqdm


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

def annotate_motion_on_frames(input_folder):
    output_path = f"{input_folder}_annotion"
    os.makedirs(output_path, exist_ok=True)
    for root, dirs, files in os.walk(input_folder):
        for dir in tqdm(dirs, desc="Processing directories"):
            dir_path = os.path.join(root, dir)
            frame_files = sorted(os.listdir(dir_path))
            frame_files = [f for f in frame_files if f.lower().endswith((".jpg", ".png"))]

            # Create output directory for this subdirectory
            output_dir = os.path.join(output_path, dir)
            os.makedirs(output_dir, exist_ok=True)

            for i in tqdm(range(1, len(frame_files)), desc=f"Processing frames in {dir}", leave=False):
                prev_path = os.path.join(dir_path, frame_files[i - 1])
                curr_path = os.path.join(dir_path, frame_files[i])
                prev_gray = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
                curr_gray = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
                curr_color = cv2.imread(curr_path)  # color image for annotation

                if prev_gray is None or curr_gray is None or curr_color is None:
                    continue

                avg_motion = compute_avg_motion(prev_gray, curr_gray)
                text = f"Avg Motion: {avg_motion:.4f}"
                annotated = overlay_text_on_frame(curr_color, text)

                file_output_path = os.path.join(output_dir, frame_files[i])
                cv2.imwrite(file_output_path, annotated)
                print(f"Saved: {file_output_path}")

    return
def still_treat(dir_path: str, frame_files: list, half_move_files: list,save_dir: str) -> None:
    """
    Move files from dir_path to output_dir if they are not in half_move_files.
    Ensures max_file is included in half_move_files before moving.
    """

    for file in frame_files:

        src = os.path.join(dir_path, file)
        

        if file not in half_move_files:
            txt_file = os.path.join(save_dir, "still.txt")

        else:
            txt_file = os.path.join(save_dir, "bad.txt")

        with open(txt_file, "a") as file:
            file.write(f"{src}\n")

def optical_flow_statistics(avg_motion_list: list ,save_dir: str) -> None:

    moving_avg_motion_list = avg_motion_list[0]
    still_avg_motion_list = avg_motion_list[1]

    with open(os.path.join(save_dir, "optical_flow_statistics.txt"), "w") as file:
        #write title
        date = datetime.now().strftime('%Y-%m-%d')
        file.write(f"Optical Flow Statistics\n date: {date}\n")
        file.write("=====================\n")
        
        #write the number of rows on save dir still.txt and bad.txt
        if os.path.exists(os.path.join(save_dir, "still.txt")):
            still_count = len(open(os.path.join(save_dir, "still.txt")).readlines())
            file.write(f"Still Count: {still_count}\n")
        if os.path.exists(os.path.join(save_dir, "move.txt")):
            move_count = len(open(os.path.join(save_dir, "move.txt")).readlines())
            file.write(f"move Count: {move_count}\n")
        

        #write the moving avg motion and still avg motion
        if len(moving_avg_motion_list) != 0:
            file.write(f"Moving Avg Motion: {np.mean(moving_avg_motion_list):.4f}\n")
            file.write(f"Moving Std Motion: {np.std(moving_avg_motion_list):.4f}\n")
        else:
            file.write("Moving Avg Motion: 0.0000\n")
            file.write("Moving Std Motion: 0.0000\n")

        if len(still_avg_motion_list) != 0:
            file.write(f"Still Avg Motion: {np.mean(still_avg_motion_list):.4f}\n")
            file.write(f"Still Std Motion: {np.std(still_avg_motion_list):.4f}\n")
        else:
            file.write("Still Avg Motion: 0.0000\n")
            file.write("Still Std Motion: 0.0000\n")


    

def is_moving(dir_path,  motion_threshold, save_dir,avg_motion_list):

    half_move_files = []
    moving_avg_motion_list = []
    still_avg_motion_list = []
    max_mov = 0
    avg_motion = 0
    folder_mean = 0
    max_file = 0 
    

    frame_files = sorted(os.listdir(dir_path))
    frame_files = [f for f in frame_files if f.lower().endswith((".jpg", ".png"))]
    
    for i in tqdm(range(1, len(frame_files)), desc=f"Processing frames in {dir_path}", leave=False):
        prev_dir = os.path.join(dir_path, frame_files[i - 1])
        curr_dir = os.path.join(dir_path, frame_files[i])
        prev = cv2.imread(prev_dir, cv2.IMREAD_GRAYSCALE)
        curr = cv2.imread(curr_dir, cv2.IMREAD_GRAYSCALE)

        if prev is None or curr is None:
            continue

        avg_motion = compute_avg_motion(prev, curr)

        # if avg_motion > 0.1:
        #     if frame_files[i - 1] not in half_move_files:
        #         half_move_files.append(frame_files[i - 1])
        #     half_move_files.append(frame_files[i])

        if avg_motion > max_mov:
            max_mov = avg_motion
            max_file = i

        folder_mean += avg_motion


    folder_mean = folder_mean/(len(frame_files))
    if folder_mean > motion_threshold:
        avg_motion_list[0].append(folder_mean)
        for  i in range(1, len(frame_files)):
            if not frame_files[i] in half_move_files:
                half_move_files.append(frame_files[i])

    else:
        avg_motion_list[1].append(folder_mean)


    if max_file not in half_move_files:
        half_move_files.append(frame_files[i])
        half_move_files.append(frame_files[i-1])


    still_treat(dir_path, frame_files, half_move_files,save_dir)

    return avg_motion_list

def check_moving_folder(folder_path, motion_threshold,save_dir):
    
    avg_motion_list = [[],[]]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    unique_parents = set()
    for file in Path(folder_path).rglob("*.jpg"):
        unique_parents.add(file.parent)

    for parent in tqdm(unique_parents, desc="Processing directories"):
        dir_path = os.path.join(folder_path, parent)
        avg_motion_list = is_moving(dir_path, motion_threshold,save_dir,avg_motion_list)
    
    optical_flow_statistics(avg_motion_list,save_dir)

if __name__ == "__main__":
    for cam in ['160','161','162','164','165','200','201','202','203','204','205','210', '211','212','213','214','215']:
        input_folder = f"/Nagmash_GZ_july/frames/{cam}/2025/07"
        save_dir = f"/Nagmash_GZ_july/frames/{cam}/2025/07_optical_flow"
        motion_threshold = 4
    check_moving_folder(input_folder, motion_threshold,save_dir)
   