#!/usr/bin/env python3

import numpy as np
import cv2
import os
import time
from display import Display2D
from collections import defaultdict, namedtuple

class ImageReader:
  def __init__(self, ids):
    self.ids = ids
    self.idx = 0

  def read(self, path):
    img = cv2.imread(path)
    return img

  def __getitem__(self, idx):
    self.idx = idx
    img = self.read(self.ids[idx])
    return np.array(img)

  def __len__(self):
    return len(self.ids)

class KittiDataReader:
  def __init__(self, path):
    Cam = namedtuple('cam', 'fx fy cx cy width height')
    self.cam = Cam(718.856, 718.856, 607.1928, 185.2157, 1241, 376)

    # expand to absolute path 
    path = os.path.expanduser(path)

    self.left = ImageReader(self.listdir(os.path.join(path, 'image_0')))
    self.right = ImageReader(self.listdir(os.path.join(path, 'image_1')))

    assert len(self.left) == len(self.right)

  def listdir(self, dir):
    files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
    return [os.path.join(dir, _) for _ in self.sort(files)]

  def sort(self, xs):
    return sorted(xs, key=lambda x:float(x[:-4]))

  def __getitem__(self, idx):
    return self.left[idx]

  def __len__(self):
    return len(self.left)


if __name__ == '__main__':
  display = Display2D(1241, 376)

  t = KittiDataReader('~/Desktop/pyslam/dataset/sequences/00')

  for i in range(len(t.left)):
    path = t.left.ids[i]
    img = np.array(cv2.imread(path))
    display.paint(img)

