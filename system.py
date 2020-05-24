#!/usr/bin/env python3

import cv2
import sys
import argparse

import numpy as np
from display import Display2D, Display3D
from data_manager import KittiDataReader
from feature import Feature
from camera import Camera 
from frame import Frame 
from config import KittiConfig

sys.path.append('lib')
import g2o


class PYSLAM:
  def __init__(self, config):
    self.config = config

    self.initialized = False

    self.reference = None
    self.last = None
    self.current = None

  def initialize(self, frame):
    print('INITIALIZING')
    self.reference = frame 
    self.last = frame 
    self.current = frame 

    self.initialized = True

  def observe(self, frame):
    print('OBSERVING')
    self.current = frame

    idx1s, idx2s, matches = frame.get_matches(self.reference)

    frame.estimate_pose(matches)

    img = frame.annotate(self.reference)

    #print('Frame ID: %d - Num Matches: %d' % (self.current.idx, len(matches)))

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
  parser.add_argument('--path', type=str, help='path to dataset', default='~/Desktop/pyslam/dataset/sequences/00')
  args = parser.parse_args()

  # Kitti configuration
  config = KittiConfig()
 
  # Kitti dataset loader
  dataset = KittiDataReader(args.path)

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
  #viewer = Display3D(slam, config)
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

