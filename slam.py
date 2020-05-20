#!/usr/bin/env python3

import sys
sys.path.append("lib")

import cv2
import pygame
import numpy as np
from pygame.locals import DOUBLEBUF
from display import Display2D
from frame import Frame, get_matches
from helpers import FtoRt
import pypangolin as pangolin

class Map:
  def __init__(self):
    self.frames = []
    self.frame_idx = 0

  def add_frame(self, f):
    self.frame_idx += 1
    self.frames.append(f)
    return self.frame_idx

class SLAM:
  def __init__(self, W, H, K):
    self.W, self.H = W, H
    # map
    self.map = Map()
    # camera matrix
    self.K = K

  def process_frame(self, img):
    frame = Frame(img, self.map, self.K)
    self.map.add_frame(frame)
    if frame.ID == 0:
      return img

    f1 = self.map.frames[-1]
    f2 = self.map.frames[-2]

    matches, F = get_matches(f1, f2)
    R, t = FtoRt(F, self.K)

    for p1, p2 in matches:
      u1, v1 = (p1[0], p1[1])
      u2, v2 = (p2[0], p2[1])

      cv2.circle(img, (u1, v2), color=(0, 255, 0), radius=2)
      cv2.circle(img, (u2, v2), color=(0, 255, 255), radius=1)
      cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    return img

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('%s <video.mp4>' % sys.argv[0])
    exit(-1)

  cap = cv2.VideoCapture(sys.argv[1])

  # frame props
  W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2

  # focal length
  F = 1

  # camera matrix
  # adjust mapping: img coords have origin at center, digital have origin bottom-left
  K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

  slam = SLAM(W, H, K)
  disp = Display2D(W, H)

  while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (W, H))

    if ret == True:
      img = slam.process_frame(frame)
      disp.paint(img)
    else:
      break
