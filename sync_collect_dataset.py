"""
  __  __                   _   _             
 |  \/  | ___  _ __   ___ | \ | | __ ___   __
 | |\/| |/ _ \| '_ \ / _ \|  \| |/ _` \ \ / /
 | |  | | (_) | | | | (_) | |\  | (_| |\ V / 
 |_|  |_|\___/|_| |_|\___/|_| \_|\__,_| \_/  
Copyright (c) 2023 Nate Simon
License: MIT
Authors: Nate Simon and Anirudha Majumdar, Princeton University
Project Page: https://natesimon.github.io/mononav

The purpose of this script is to collect a custom dataset for the MonoNav demo.
This script saves images and poses from a Crazyflie to file.
Those images and poses can then be used in the demo (depth estimation, fusion, and simulated planning).

"""
from enum import Enum

import cv2
import numpy as np
import time
import os
import time
import pykinect_azure as pykinect
from scipy.spatial.transform import Rotation as Rotation


# For Craziflie logging
import tellopy

from utils.utils import VideoCapture, get_rootdir, \
    RealsenseVideoCapture, AzureVideoCapture, TelloVideoCapture

'''
This file collects the following synchronized images:
- Tello (front-facing camera)
- Tello pose
- Timestamps
'''

# for Tello position data
def log_data_handler(event, sender, data, **args):
    '''
    EVENT_LOG_DATA:
    mvo: 
    pos_x
    pos_y
    pos_z
    vel_x
    vel_y
    vel_z
    imu:
    acc_x
    acc_y
    acc_z
    gyro_x
    gyro_y
    gyro_z
    q0
    q1
    q2
    q3
    vg_x
    vg_y
    vg_z
    vo: # Visual Odometry (more robust than MVO)
    pos_x
    pos_y
    pos_z
    vel_x
    vel_y
    vel_z
    height: # uSonic sensor
    height
    '''
    drone = sender
    global _x, _y, _z
    global _q0, _q1, _q2, _q3
    if event is drone.EVENT_LOG_DATA:
        _x = data.vo.pos_x
        _y = data.vo.pos_y
        _z = data.height.height
        _q0 = data.imu.q0
        _q1 = data.imu.q1
        _q2 = data.imu.q2
        _q3 = data.imu.q3

# make diff GT sources
class GTSource(Enum):
    AZURE = "azure"
    REALSENSE = "realsense"

# def ground truth source (azure kinect or D415 realsense cam)
gt_source = GTSource.AZURE

def main():
    
    # set up Tello connection and logging
    drone = tellopy.Tello()
    drone.subscribe(drone.EVENT_LOG_DATA, log_data_handler)
    drone.connect()
    drone.wait_for_connection(60.0)

    # Make directories for data
    save_dir = 'data/synced-trial-' + time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    # subdirectories for GT readings and CF images
    tello_img_dir = os.path.join(save_dir, "tello-rgb-images")
    tello_pose_dir = os.path.join(save_dir, "tello-poses")
    gt_rgb_dir = os.path.join(save_dir, gt_source.value + "-rgb-images")
    gt_depth_dir = os.path.join(save_dir, gt_source.value + "-depth-images")

    # make directories if they don't already exist
    os.mkdir(tello_img_dir) if not os.path.exists(tello_img_dir) else None
    os.mkdir(tello_pose_dir) if not os.path.exists(tello_pose_dir) else None
    os.mkdir(gt_rgb_dir) if not os.path.exists(gt_rgb_dir) else None
    os.mkdir(gt_depth_dir) if not os.path.exists(gt_depth_dir) else None

    # log dir location
    print("Saving files to: " + save_dir)

    # Drone object
    # Initialize the low-level drivers

    # Camera Object
    cap = TelloVideoCapture(drone)

    # get GT source capture
    if gt_source == GTSource.REALSENSE:
        gt_cap = RealsenseVideoCapture('RGB Camera')
    else:
        gt_cap = AzureVideoCapture()

    # Initialize counter for images
    frame_number = 0
    xyz_start = np.array([_y, _z, _x])
    # main loop
    while True:
        # print frame number
        print("Frame Number %d" % frame_number)

        # read depth, color images from gt source and read cf cap as well
        try:
            depth_image, color_image = gt_cap.read()
        except RuntimeError as gt_err:
            print(gt_err)
            continue

        tello_rgb = cap.read()  # should have fixed lag
        tello_rgb_numpy = np.array(tello_rgb)
        tello_rgb_opencv = cv2.cvtColor(tello_rgb_numpy, cv2.COLOR_RGB2BGR)
        # Map the depth image np array to a visual map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Save images and depth array
        cv2.imwrite(tello_img_dir + "/tello_frame-%06d.rgb.jpg" % (frame_number), tello_rgb_opencv)
        xyz_now = np.array([_y, _z, _x])  # Convert to TSDF frame
        xyz = xyz_now - xyz_start

        # Get orientation
        q = np.array([_q1, _q2, _q3, _q0])  # Store quaternion in scalar-last format (used by scipy)

        # Convert to rotation matrix
        r = Rotation.from_quat(q)
        R = r.as_matrix()
        # Find rotation matrix relative to original
        # R_diff = R_original_inv @ R 
        # R_diff = R
        print(r.as_euler('xyz', degrees=True))
        M_change = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Change from Tello to TSDF frame
        R = M_change @ R @ M_change.T  # Convert to TSDF frame

        # Correct for Tello camera tilt (-10 degrees about TSDF x-axis)
        camera_tilt = -10 * np.pi / 180  # 10 degrees
        R_camera_tilt = np.array([[1, 0, 0], [0, np.cos(camera_tilt), -np.sin(camera_tilt)],
                                    [0, np.sin(camera_tilt), np.cos(camera_tilt)]])
        R = R @ R_camera_tilt

        # Homogeneous matrix
        np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format}) 
        Hmtrx = np.hstack((R, xyz_now.reshape(3, 1)))
        Hmtrx = np.vstack((Hmtrx, np.array([0, 0, 0, 1])))
        camera_position = Hmtrx  # get camera position immediately
        np.savetxt(tello_pose_dir + "/tello_frame-%06d.pose.txt" % (frame_number), camera_position)
        # Save ground truth results also
        cv2.imwrite(gt_rgb_dir + "/%s_frame-%06d.color.jpg" % (gt_source.value, frame_number), color_image)
        cv2.imwrite(gt_depth_dir + "/%s_frame-%06d.depth.jpg" % (gt_source.value, frame_number), depth_colormap)
        np.save(gt_depth_dir + "/%s_frame-%06d.depth.npy" % (gt_source.value, frame_number), depth_image)

        # Update counter
        frame_number += 1

        # Show images
        # cv2.imshow('tello', tello_rgb)
        cv2.waitKey(1)  # 1 ms

        # OPTIONAL: show color and depth images from ground truth as well
        # cv2.imshow("azure", color_image)
        # cv2.waitKey(1)
        # cv2.imshow("depth", depth_colormap)
        # cv2.waitKey(1)

        # check exit condition
        if chr(cv2.waitKey(1) & 255) == 'q':
            break

        # pause between frames
        time.sleep(0.2)

    # clear caps and remove windows
    cap.cap.release()
    cv2.destroyAllWindows()
    drone.quit()
    
if __name__ == '__main__':
    main()
