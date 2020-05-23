import cv2


class Frame:
  def __init__(self, idx, pose, feature, cam):
    self.idx = idx
    self.pose = pose
    self.feature = feature 
    self.cam = cam
    self.image = feature.image
 
  # computes matches between current frame and reference frame
  def get_matches(self, reference_frame):
    E = self.feature.get_matches(
        reference_frame.feature.keypoints, 
        reference_frame.feature.descriptors)

    print(E)

    return E


