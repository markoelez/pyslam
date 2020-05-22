#!/usr/bin/env python3

import sys
sys.path.append("lib")

import cv2
import pygame
import numpy as np
from pygame.locals import DOUBLEBUF
from display import Display2D
from frame_old import Frame, get_matches, world_to_camera, E_to_Rt, triangulate

import pypangolin as pangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue

class Map:
  def __init__(self):
    self.frames = []
    self.points = []
    self.state = None

    self.viewer_init()

  def viewer_init(self):
    pangolin.CreateWindowAndBind("main", 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    self.handler = pangolin.Handler3D(self.scam)

    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(
      pangolin.Attach(0.0), pangolin.Attach(1.0), pangolin.Attach(0.0), pangolin.Attach(1.0), -640.0 / 480.0)
    self.dcam.SetHandler(self.handler)
    self.dcam.Activate()
    '''
    # Create Interactive View in window
    self.dcam = pangolin.CreateDisplay()
    print(dir(pangolin.View))
    #view = pangolin.View.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    #self.dcam.SetBounds(view)
    self.dcam.SetHandler(self.handler)
    '''

  def viewer_refresh(self):
    if self.state == None:
      return
    # turn state into points
    ppts = np.array([d[:3, 3] for d in self.state[0]])
    spts = np.array(self.state[1])

    print(ppts)

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    self.dcam.Activate(self.scam)

    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawPoints(ppts)

    gl.glPointSize(2)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawPoints(spts)

    pangolin.FinishFrame()

  def display(self):
    poses, pts = [], []
    for f in self.frames:
      poses.append(f.pose)
    for p in self.points:
      pts.append(p.pt)
    self.state = poses, pts 
    self.viewer_refresh()

class Point:
  def __init__(self, mapp, loc):
    self.pt = loc
    self.frames = []
    self.idxs = []

    self.id = len(mapp.points)
    mapp.points.append(self)

  def add_observation(self, frame, idx):
      self.frames.append(frame)
      self.idxs.append(idx)

class SLAM:
  def __init__(self, W, H, K):
    self.W, self.H = W, H
    # map
    self.mapp = Map()
    # camera matrix
    self.K = K

  def process_frame(self, img):
    frame = Frame(img, self.mapp, self.K)
    if frame.ID == 0:
      return img

    f1 = self.mapp.frames[-1]
    f2 = self.mapp.frames[-2]

    idx1s, idx2s, E = get_matches(f1, f2)

    Rt = E_to_Rt(E)
    f1.pose = np.dot(Rt, f2.pose)

    # homogeneous 3-D coords
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1s], f2.pts[idx2s])
    pts4d /= pts4d[:, 3:]
    #print(pts4d)

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

    for i, p in enumerate(pts4d):
      if not good_pts4d[i]:
        continue
      pt = Point(self.mapp, p)
      pt.add_observation(f1, idx1s[i])
      pt.add_observation(f2, idx2s[i])
    '''
    for p1, p2 in zip(f1.pts[idx1s], f2.pts[idx2s]):
      u1, v1 = world_to_camera(self.K, p1)
      u2, v2 = world_to_camera(self.K, p2)

      cv2.circle(img, (u1, v2), color=(0, 255, 0), radius=2)
      cv2.circle(img, (u2, v2), color=(0, 255, 255), radius=1)
      cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))
    '''
    self.mapp.display()

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
  #disp = Display2D(W, H)

  while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (W, H))

    if ret == True:
      img = slam.process_frame(frame)
      #disp.paint(img)
    else:
      break
