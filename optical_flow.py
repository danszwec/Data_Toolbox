import os
from pathlib import Path
import cv2
from tqdm import tqdm
import yaml
from utils import compute_avg_motion, optical_flow_statistics





def classify_frames_by_movement(dir_path: str, frame_files: list, move_files: list,save_dir: str) -> None:
    """
    Move files from dir_path to output_dir if they are not in move_files.
    Ensures max_file is included in move_files before moving.
    """

    for file in frame_files:

        src = os.path.join(dir_path, file)
        

        if file not in move_files:
            txt_file = os.path.join(save_dir, "still.txt")

        else:
            txt_file = os.path.join(save_dir, "move.txt")

        with open(txt_file, "a") as file:
            file.write(f"{src}\n")


    return

def is_moving(dir_path: str,  motion_threshold: float, save_dir: str,moving_motion_list: list,still_motion_list: list,classify_by: str) -> list:
    """
    Check if the directory is moving based on the average motion.
    Args:
        dir_path: str, the path to the directory
        motion_threshold: float, the threshold for the average motion
        save_dir: str, the path to the save directory
        avg_motion_list: list, the list of average motion
    Returns:
        avg_motion_list: list, the list of average motion
    """

    #initialize variables
    move_files = []
    moving_avg_motion_list = []
    still_avg_motion_list = []
    max_mov = 0
    avg_motion = 0
    folder_mean = 0
    max_file = 0 
    
    #get the frame files
    frame_files = sorted(os.listdir(dir_path))
    frame_files = [f for f in frame_files if f.lower().endswith((".jpg", ".png"))]
    
    #compute the average motion for each frame
    for i in tqdm(range(1, len(frame_files)), desc=f"Processing frames in {dir_path}", leave=False):
        prev_dir = os.path.join(dir_path, frame_files[i - 1])
        curr_dir = os.path.join(dir_path, frame_files[i])
        prev = cv2.imread(prev_dir, cv2.IMREAD_GRAYSCALE)
        curr = cv2.imread(curr_dir, cv2.IMREAD_GRAYSCALE)

        #check if the frames are valid
        if prev is None or curr is None:
            continue

        avg_motion = compute_avg_motion(prev, curr)

        #if the classify by is frame, add the avg motion to the moving avg motion list
        if classify_by == "folder":

            #update the max motion
            if avg_motion > max_mov:
                max_mov = avg_motion
                max_file = i

            #update the folder mean
            folder_mean += avg_motion


            #compute the folder mean
            folder_mean = folder_mean/(len(frame_files))

            #if the folder mean is greater than the motion threshold, add the folder mean to the moving avg motion list
            if folder_mean > motion_threshold:
                moving_motion_list.append(folder_mean)
                for  i in range(1, len(frame_files)):
                    if not frame_files[i] in move_files:
                        move_files.append(frame_files[i])

            #if the folder mean is less than the motion threshold, add the folder mean to the still avg motion list
            else:
                still_motion_list.append(folder_mean)

            #if the max file is not in the move files, add the max file to the move files instead of the still files
            if max_file not in move_files:
                move_files.append(frame_files[i])
                move_files.append(frame_files[i-1])

        elif classify_by == "frame":

            #if the avg motion is greater than the motion threshold, add the avg motion to the moving avg motion list
            if avg_motion > motion_threshold:
                move_files.append(frame_files[i])

                if frame_files[i-1] not in move_files:
                    move_files.append(frame_files[i-1])

                moving_motion_list.append(avg_motion)

            #if the avg motion is less than the motion threshold, add the avg motion to the still avg motion list
            else:
                still_motion_list.append(avg_motion)

            
         
    classify_frames_by_movement(dir_path, frame_files, move_files,save_dir)

    return moving_motion_list,still_motion_list

def main(config: dict) -> None:
    """
    Check if the folder is moving based on the average motion.
    Args:
        config: dict, the config dictionary
    Returns:
        None
    """
    #unpack the config
    input_path = config["input_path"]
    folders = config["folders"]
    motion_threshold = config["motion_threshold"] 
    classify_by = config["classify_by"]

    for folder in tqdm(os.listdir(input_path), desc=f"Processing folders in {input_path}"):
        folder_path = os.path.join(input_path, folder)
        save_dir = f"{folder_path}_optical_flow"

        #initialize variables
        moving_motion_list = []
        still_motion_list = []

        #create the save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #get the unique parents
        unique_parents = set()
        for file in Path(folder_path).rglob("*.jpg"):
            unique_parents.add(file.parent)

        #process the unique parents
        for parent in tqdm(unique_parents, desc=f"Processing directories in {folder_path}"):
            dir_path = os.path.join(folder_path, parent)
            moving_motion_list,still_motion_list = is_moving(dir_path, motion_threshold,save_dir,moving_motion_list,still_motion_list,classify_by)

        #save the statistics
        optical_flow_statistics(moving_motion_list,still_motion_list,save_dir)

    return


if __name__ == "__main__":
    #load the config
    with open("cfg.yaml", "r") as file:
        config = yaml.safe_load(file)

    main(config)
   