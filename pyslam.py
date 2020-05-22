#!/usr/bin/env python3

import cv2
import sys
import numpy as np
from display import Display2D
from data_manager import KittiDataReader
from feature import Feature
from camera import Camera 
from frame import Frame 

sys.path.append('lib')
import g2o


if __name__ == '__main__':
  display = Display2D(1241, 376)

  dataset = KittiDataReader('~/Desktop/pyslam/dataset/sequences/00')
  camera = Camera(
      dataset.cam.fx,
      dataset.cam.fy,
      dataset.cam.cx,
      dataset.cam.cy,
      dataset.cam.width,
      dataset.cam.height)

  frames = []

  for i in range(len(dataset)):
    img = dataset[i]

    feature = Feature(img)
    feature.extract()

    frame = Frame(i, g2o.Isometry3d(), feature, camera)

    if len(frames) > 1:
      f1 = frames[-1]

      m = f1.feature.get_matches(frame.feature.descriptors)
      for (p1, p2) in m:
        print(p1, p2)
      
    frames.append(frame)
    
    img = frame.feature.draw_keypoints()

    display.paint(img)

