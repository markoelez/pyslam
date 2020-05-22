import cv2
import numpy as np
from collections import defaultdict

class Feature:
  def __init__(self, image):
    self.image = image
    self.width, self.height = image.shape[:2]

    self.detector = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=1, edgeThreshold=31)
    self.extractor = self.detector
    self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    self.distance = 25

  def extract(self):
    self.keypoints = self.detector.detect(self.image)
    self.keypoints, self.descriptors = self.extractor.compute(self.image, self.keypoints)
    self.unmatched = np.ones(len(self.keypoints), dtype=bool)

  def get_matches(self, descriptors):
    matches = dict()
    matched_by = self.matcher.match(np.array(descriptors), self.descriptors)
    distances = defaultdict(lambda: float('inf'))

    for m in matched_by:
      if m.distance > min(distances[m.trainIdx], self.distance):
        continue

      matches[m.trainIdx] = m.queryIdx
      distances[m.trainIdx] = m.distance

    matches = [(i, j) for j, i in matches.items()]
    return matches

  def matched_by(self, descriptors):
    matches = self.matcher.match(np.array(descriptors), unmatched_descriptors)
    return [(m, m.queryIdx, m.trainIdx) for m in matches]

  def draw_keypoints(self):
    img = cv2.drawKeypoints(self.image, self.keypoints, None, flags=0)
    return img

  def get_keypoints(self, idx):
    return self.keypoints[idx]

  def get_descriptor(self, idx):
    return self.descriptors(idx)
