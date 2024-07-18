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

# For Craziflie logging
import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

from utils.utils import reset_estimator, VideoCapture, get_crazyflie_pose, get_rootdir, \
    RealsenseVideoCapture, AzureVideoCapture, TelloVideoCapture

'''
This file collects the following synchronized images:
- Crazyflie (front-facing camera)
- Crazyflie pose
- Timestamps
'''


# make diff GT sources
class GTSource(Enum):
    AZURE = "azure"
    REALSENSE = "realsense"


# def ground truth source (azure kinect or D415 realsense cam)
gt_source = GTSource.REALSENSE

# set up CF connection and logging
URI = uri_helper.uri_from_env(default='radio://0/1/2M/E7E7E7E7E7')
logging.basicConfig(level=logging.ERROR)
camera_num = 1

# Make directories for data
save_dir = 'data/synced-trial-' + time.strftime("%Y-%m-%d-%H-%M-%S")
os.mkdir(save_dir) if not os.path.exists(save_dir) else None

# subdirectories for GT readings and CF images
crazyflie_img_dir = os.path.join(save_dir, "crazyflie-rgb-images")
crazyflie_pose_dir = os.path.join(save_dir, "crazyflie-poses")
gt_rgb_dir = os.path.join(save_dir, gt_source.value + "-rgb-images")
gt_depth_dir = os.path.join(save_dir, gt_source.value + "-depth-images")

# make directories if they don't already exist
os.mkdir(crazyflie_img_dir) if not os.path.exists(crazyflie_img_dir) else None
os.mkdir(crazyflie_pose_dir) if not os.path.exists(crazyflie_pose_dir) else None
os.mkdir(gt_rgb_dir) if not os.path.exists(gt_rgb_dir) else None
os.mkdir(gt_depth_dir) if not os.path.exists(gt_depth_dir) else None

# log dir location
print("Saving files to: " + save_dir)

# Drone object
# Initialize the low-level drivers
cflib.crtp.init_drivers()

# Set up log conf
logstate = LogConfig(name='state', period_in_ms=10)
logstate.add_variable('stateEstimate.x', 'float')
logstate.add_variable('stateEstimate.y', 'float')
logstate.add_variable('stateEstimate.z', 'float')
logstate.add_variable('stateEstimate.roll', 'float')
logstate.add_variable('stateEstimate.pitch', 'float')
logstate.add_variable('stateEstimate.yaw', 'float')

# connect to CF
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    cf = scf.cf
    # reset pose estimation
    reset_estimator(cf)

    # Camera Object
    cap = VideoCapture(camera_num)

    # get GT source capture
    if gt_source == GTSource.REALSENSE:
        gt_cap = RealsenseVideoCapture('RGB Camera')
    else:
        gt_cap = AzureVideoCapture()

    # Initialize counter for images
    frame_number = 0

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

        crazyflie_rgb = cap.read()  # should have fixed lag

        # Map the depth image np array to a visual map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Save images and depth array
        cv2.imwrite(crazyflie_img_dir + "/crazyflie_frame-%06d.rgb.jpg" % (frame_number), crazyflie_rgb)
        camera_position = get_crazyflie_pose(scf, logstate)
        np.savetxt(crazyflie_pose_dir + "/crazyflie_frame-%06d.pose.txt" % (frame_number), camera_position)
        # Save ground truth results also
        cv2.imwrite(gt_rgb_dir + "/%s_frame-%06d.color.jpg" % (gt_source.value, frame_number), color_image)
        cv2.imwrite(gt_depth_dir + "/%s_frame-%06d.depth.jpg" % (gt_source.value, frame_number), depth_colormap)
        np.save(gt_depth_dir + "/%s_frame-%06d.depth.npy" % (gt_source.value, frame_number), depth_image)

        # Update counter
        frame_number += 1

        # Show images
        cv2.imshow('crazyflie', crazyflie_rgb)
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
