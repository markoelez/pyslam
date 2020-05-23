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
  #display = Display3D(slam, config)
  display = Display2D(dataset.cam.width, dataset.cam.height)

  frames = []
  for i in range(len(dataset)):
    img = dataset[i]

    feature = Feature(img, config, camera)
    feature.extract()

    frame = Frame(i, np.eye(4), feature, camera)
    if len(frames) > 1:
      reference_frame = frames[-1]
      E = reference_frame.get_matches(reference_frame)
      
    frames.append(frame)
    
    img = frame.feature.draw_keypoints()

    #print('Frame ID: %d - Num Matches: %d' % (i, len(m)))
    display.paint(img)

