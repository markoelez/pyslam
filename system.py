#!/usr/bin/env python3

import cv2
import sys
import argparse

import numpy as np
from display import Display2D
from loader import KittiLoader 
from feature import Feature
from camera import Camera 
from frame import Frame 
from config import KittiConfig, VideoConfig

sys.path.append('lib')


class PYSLAM:
  def __init__(self, config):
    self.config = config

    self.initialized = False

    self.reference = None
    self.last = None
    self.current = None
    self.points = [] 

  def initialize(self, frame):
    self.reference = frame 
    self.last = frame 
    self.current = frame 

    self.initialized = True

  def observe(self, frame):
    self.current = frame

    idx1s, idx2s, matches = frame.get_matches(self.reference)

    pose = frame.estimate_pose(self.reference)
    
    img = frame.annotate(self.reference)
    points3D = frame.triangulate_points(self.reference)

    good_points3D = (np.abs(points3D[:, 3]) > 0.005) & (points3D[:, 2] > 0)
    
    for i, pt in enumerate(points3D):
      if not good_points3D[i]:
        continue
      self.points.append(pt)

    print('Frame ID: {} - Num Matches: {} - Points: {}'.format(self.current.idx, len(matches), points3D[:5]))

    # update last
    self.reference = self.current
    self.last = self.current

    return img

  def is_initialized(self):
    return self.initialized

  def get_current_image(self):
    return self.current.image


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--type', type=str, help='input type: KITTI or VIDEO', default='KITTI')
  parser.add_argument('--path', type=str, help='path to input data', default='dataset/sequences/00')
  args = parser.parse_args()

  if args.type == 'KITTI':
    # Kitti configuration
    config = KittiConfig()
 
    # Kitti dataset loader
    dataset = KittiLoader(args.path)

    # camera object
    camera = Camera(
        dataset.cam.fx,
        dataset.cam.fy,
        dataset.cam.cx,
        dataset.cam.cy,
        dataset.cam.width,
        dataset.cam.height)

    # global SLAM object
    slam = PYSLAM(config)

    # display 
    viewer = Display2D(slam, config)

    frames = []
    for i in range(len(dataset)):
      img = dataset[i]

      # extract features
      feature = Feature(img, config, camera)
      feature.extract()
      
      # init frame
      frame = Frame(i, np.eye(4), feature, camera)

      if slam.is_initialized():
        img = slam.observe(frame)
      else:
        slam.initialize(frame)

      viewer.update()

  if args.type == 'VIDEO':
    cap = cv2.VideoCapture(args.path)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2
    
    # initialize config with image dimensions
    config = VideoConfig(W, H)

    camera = Camera(
        500,
        500,
        W//2,
        H//2,
        W//2,
        H//2)

    # global SLAM object
    slam = PYSLAM(config)

    # display 
    viewer = Display2D(slam, config)

    i = 0
    while cap.isOpened():
      ret, frame = cap.read()
      frame = cv2.resize(frame, (W, H))

      if ret == True:
        # extract features
        feature = Feature(frame, config, camera)
        feature.extract()
        
        # init frame
        frame = Frame(i, np.eye(4), feature, camera)

        if slam.is_initialized():
          frame = slam.observe(frame)
        else:
          slam.initialize(frame)

        viewer.update()
      else:
        break

      i += 1


