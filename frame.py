#!/usr/bin/env python3

import cv2
import numpy as np


class Frame:
  def __init__(self, idx, pose, feature, camera):
    self.idx = idx
    self.pose = pose
    self.feature = feature 
    self.camera = camera 
    self.image = feature.image

  # computes matches between current frame and reference frame
  def get_matches(self, reference_frame):
    idx1s, idx2s, matches = self.feature.get_matches(
        reference_frame.feature.keypoints, 
        reference_frame.feature.descriptors)
    
    self.matches = matches
    self.idx1s = idx1s 
    self.idx2s = idx2s 

    return idx1s, idx2s, matches

  def estimate_pose(self, reference_frame):
    src = self.matches[:, 0]
    dst = self.matches[:, 1]

    E, mask = cv2.findEssentialMat(
        src, 
        dst, 
        self.camera.intrinsic, 
        method=cv2.RANSAC)

    ret, R, t, mask = cv2.recoverPose(E, src, dst, self.camera.intrinsic)

    # combine into pose
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 2:] = t

    self.pose = np.dot(Rt, reference_frame.pose)
    #print('\nUpdated pose: \n{}\n'.format(Rt))

    return self.pose 

  def triangulate_points(self, reference_frame):
    matches = self.get_matches(reference_frame)

    kps1 = np.array([(int(p.pt[0]), int(p.pt[1])) for p in self.feature.keypoints[self.idx1s]])
    kps2 = np.array([(int(p.pt[0]), int(p.pt[1])) for p in reference_frame.feature.keypoints[self.idx2s]])

    points4D = self.triangulate(self.pose, reference_frame.pose, kps1, kps2)

    points4D /= points4D[:, 3:]
    points3D = points4D#points4D[:, :3] 

    return points3D

  def triangulate(self, pose1, pose2, points1, points2):
    ret = np.zeros((points1.shape[0], 4))
    for i, p in enumerate(zip(points1, points2)):
      A = np.zeros((4,4))
      A[0] = p[0][0] * pose1[2] - pose1[0]
      A[1] = p[0][1] * pose1[2] - pose1[1]
      A[2] = p[1][0] * pose2[2] - pose2[0]
      A[3] = p[1][1] * pose2[2] - pose2[1]
      _, _, vt = np.linalg.svd(A)
      ret[i] = vt[3]
    return ret

  def update_pose(self, pose):
    self.pose = pose

  def annotate(self, reference_frame):
    img = self.feature.draw_keypoints((255, 0, 0))
    img = reference_frame.feature.draw_keypoints((0, 255, 0), img)

    for p1, p2 in self.matches:
      cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255))

    return img
