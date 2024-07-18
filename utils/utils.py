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

Helper functions for the MonoNav project.
Functionality should be concentrated here and shared between the scripts.

"""
import math
import time
import warnings
from typing import List, Tuple, Any, Union

import PIL
import cv2 as cv2
import numpy as np
import tellopy
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial import distance
import os
import open3d as o3d
import open3d.core as o3c
import copy
import yaml, json
from PIL import Image
import PIL
#import pyrealsense2 as rs
import av
import pykinect_azure as pykinect

# For bufferless video capture
import queue, threading

# For stabilizer type enum
from enum import Enum
import warnings

# For rootdir util function
from pathlib import Path


class VideoCapture:
    def __init__(self, name):
        """
        Bufferless VideoCapture, courtesy of Ulrich Stern (https://stackoverflow.com/a/54577746)
        Otherwise, a lag builds up in the video stream.
        """

        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


# Modified version for Realsense D415 Camera
class RealsenseVideoCapture:
    def __init__(self, name):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == name:
                found_rgb = True
                break
        if not found_rgb:
            raise Exception("No Intel RealSense Camera found!")

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        self.depth_frame = None
        self.color_frame = None

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        print("Pausing for RealSense stream initialization...")
        time.sleep(1.0)
        print("RealSense stream ready!")

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame_temp = frames.get_depth_frame()
            color_frame_temp = frames.get_color_frame()
            # Check if valid frames, skip update if frames invalid
            if not depth_frame_temp or not color_frame_temp:
                continue
            # Update most recent frames
            self.depth_frame = depth_frame_temp
            self.color_frame = color_frame_temp

    # return np array of images - depth, color
    def read(self):
        return np.asanyarray(self.depth_frame.get_data()), np.asanyarray(self.color_frame.get_data())


class TelloVideoCapture:

    def __init__(self, drone: tellopy.Tello):
        """
        Helper function used to get the current frame from the Tello video stream.

        Args:
            drone: Tello object
        """
        # Define global variable used to carry frame data outside of thread
        self.current_frame = None

        # Camera initialization process (from TelloPy examples)
        retry = 3
        self.container = None
        while self.container is None and 0 < retry:
            retry -= 1
            try:
                self.container = av.open(drone.get_video_stream())
            except Exception as ave:
                print(ave)
                print('retry...')

        # skip first 100 frames
        self.frame_skip = 50

        # start thread
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        print("Pausing for Tello stream initialization...")
        time.sleep(0.5)

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        # Main frame updating loop.
        while True:
            for frame in self.container.decode(video=0):
                if 0 < self.frame_skip:
                    self.frame_skip -= 1
                    continue
                # Once done with buffer notify user
                elif 0 == self.frame_skip:
                    print("Tello Video Stream Ready!")
                    self.frame_skip -= 1

                # Update global frame variable with new PIL Image
                self.current_frame = frame.to_image()

    # return most recent frame
    def read(self) -> PIL.Image.Image:
        return self.current_frame


class AzureVideoCapture:

    def __init__(self):
        # Initialize the library, if the library is not found, add the library path as argument
        pykinect.initialize_libraries("Azure Kinect SDK v1.4.2/sdk/windows-desktop/amd64/release/bin/k4a.dll")
        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        # print(device_config)
        # Start device
        self.device = pykinect.start_device(config=device_config)

    def read(self):
        # Get Azure frame capture
        capture = self.device.update()
        # Get the color depth image from the capture
        ret1, depth_image = capture.get_transformed_depth_image()  # np.array, transformed by kinect azure intrinsics
        ret2, color_image = capture.get_color_image()  # np array

        if not (ret1 and ret2):
            raise RuntimeError("Error in getting Azure frame!")

        return depth_image, color_image


"""
Output Matrix Format:
| |  3x3  |  X(R) |
| |  rot  |  Y(D) |
| |  mat  |  Z(F) |
|   0 0 0     1   |
"""


def get_rootdir() -> Path:
    """
    Convenience function to get the root directory of the program

    Returns: Posix Path of the root directory of the program

    """
    # File parent is utils
    # utils parent is main Mononav dir
    return Path(__file__).parent.parent