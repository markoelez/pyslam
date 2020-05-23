#!/usr/bin/env python3

import time
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform 

def get_features(img):
  orb = cv2.ORB_create()

  # detection
  # cnvt img from 3-channels to 1-channels
  kps = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

  # (x,y) -> cv2.KeyPoint
  kps = [cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=20) for p in kps]

  # compute descriptors 
  kps, des = orb.compute(img, kps)  

  return np.array([(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]), des

def get_matches(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.des, f2.des, k=2)

  # Lowe's ratio test 
  ret = []

  idx1s = []
  idx2s = []
  for m, n in matches:
    if m.distance < 0.75*n.distance:
      p1 = f1.kps[m.queryIdx]
      p2 = f2.kps[m.trainIdx]

      if m.distance < 32:
        ret.append((p1, p2))
        idx1s.append(m.queryIdx)
        idx2s.append(m.trainIdx)

  assert len(ret) >= 8

  ret = np.array(ret)


  idx1s = np.array(idx1s)
  idx2s = np.array(idx2s)

  src = ret[:, 0]
  dst = ret[:, 1]
  print(ret)

  model, inliers = ransac((src, dst),
      EssentialMatrixTransform,
      min_samples = 8,
      residual_threshold=1, 
      max_trials=100)

  return idx1s[inliers], idx2s[inliers], np.array(model.params)

def to_homogenous(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# [u, v, 1] = K * [X, Y, Z]
def world_to_camera(K, pt):
  ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
  return int(round(ret[0])), int(round(ret[1]))

# [u, v, 1] * K' = [X, Y, Z]
def camera_to_world(Kinv, pts):
  return np.dot(Kinv, to_homogenous(pts).T).T[:, 0:2]

def E_to_Rt(E):
  diag_110 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
  W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

  U, S, V = np.linalg.svd(E)

  E = np.dot(np.dot(U, diag_110), V.transpose())

  U, S, V = np.linalg.svd(E)

  if np.linalg.det(U) < 0:
    U *= -1.0
  if np.linalg.det(V) < 0:
    V *= -1.0

  R = np.dot(np.dot(U, W), V.transpose())
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.transpose()), V.transpose())
  t = U[:, 2]

  Rt = np.eye(4)
  Rt[:3, :3] = R
  Rt[:3, 3] = t

  return Rt

def triangulate(pose1, pose2, pts1, pts2):
  return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

class Frame:
  def __init__(self, img, mapp, K):
    self.h, self.w = img.shape[0:2]
    self.pose = np.eye(4)

    # camera intrinsics
    self.K = np.array(K)
    self.Kinv = np.linalg.inv(self.K)

    # get features and descriptors 
    self.kps, self.des = get_features(img)

    # kps are in image coords (projection point), we need 3D world coords
    self.pts = camera_to_world(self.Kinv, self.kps)

    self.ID = len(mapp.frames)
    mapp.frames.append(self)

