import cv2
import numpy as np
from collections import defaultdict

class Feature:
  def __init__(self, image, config, camera):
    self.image = image
    self.camera = camera
    self.width, self.height = image.shape[:2]

    self.detector = config.feature_detector
    self.extractor = config.descriptor_extractor
    self.matcher = config.descriptor_matcher
    self.distance = config.matching_distance 

  def extract(self):
    self.keypoints = self.detector.detect(self.image)
    self.keypoints, self.descriptors = self.extractor.compute(self.image, self.keypoints)
    self.unmatched = np.ones(len(self.keypoints), dtype=bool)
  
  def get_matches(self, keypoints, descriptors):
    '''
    Computes matches between current feature object and the provided descriptors.
    Params:
      descriptors : descriptors associated with given reference frame
      keypoints: keypoints associated with given reference frame
    '''
    matches = self.matcher.knnMatch(self.descriptors, descriptors, k=2)

    ret = []
    idx1s, idx2s = [], []

    for m, n in matches:
      if m.distance < 0.75 * n.distance:
        p1 = self.keypoints[m.queryIdx]
        p2 = keypoints[m.trainIdx]
        if m.distance < self.distance:
          ret.append((p1.pt, p2.pt))
          idx1s.append(m.queryIdx)
          idx2s.append(m.trainIdx)

    assert len(ret) >= 8 # needed for RANSAC

    ret = np.array(ret)
    idx1s = np.array(idx1s)
    idx2s = np.array(idx2s)

    src = ret[:, 0]
    dst = ret[:, 1]

    E, _ = cv2.findEssentialMat(
        src, 
        dst, 
        self.camera.intrinsic, 
        method=cv2.RANSAC)

    return E

  def draw_keypoints(self):
    img = cv2.drawKeypoints(self.image, self.keypoints, None, flags=0)
    return img

  def get_keypoints(self, idx):
    return self.keypoints[idx]

  def get_descriptor(self, idx):
    return self.descriptors(idx)
