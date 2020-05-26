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
    kps = cv2.goodFeaturesToTrack(
        np.mean(self.image, axis=2).astype(np.uint8),
        3000, 
        qualityLevel=0.01, 
        minDistance=7)
    self.keypoints = [cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=20) for p in kps]

    self.keypoints, self.descriptors = self.extractor.compute(self.image, self.keypoints)
    self.keypoints = np.array(self.keypoints)

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

    return  idx1s, idx2s, ret

  def draw_keypoints(self, color, base_image=None):
    if base_image is None:
      base_image = self.image
    #img = cv2.drawKeypoints(self.image, self.keypoints, None, flags=0)
    img = base_image[:]
    for p in self.keypoints:
      cv2.circle(img, (int(p.pt[0]), int(p.pt[1])), color=color, radius=2)
    return img

  def get_keypoints(self, idx):
    return self.keypoints[idx]

  def get_descriptor(self, idx):
    return self.descriptors(idx)
