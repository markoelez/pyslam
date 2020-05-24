import cv2


class Frame:
  def __init__(self, idx, pose, feature, camera):
    self.idx = idx
    self.pose = pose
    self.feature = feature 
    self.camera = camera 
    self.image = feature.image

    self.matches = None
 
  # computes matches between current frame and reference frame
  def get_matches(self, reference_frame):
    idx1s, idx2s, matches = self.feature.get_matches(
        reference_frame.feature.keypoints, 
        reference_frame.feature.descriptors)
    
    self.matches = matches
    return idx1s, idx2s, matches

  def estimate_pose(self, matches):
    src = matches[:, 0]
    dst = matches[:, 1]

    E, mask  = cv2.findEssentialMat(
        src, 
        dst, 
        self.camera.intrinsic, 
        method=cv2.RANSAC)

    ret, R, t, mask = cv2.recoverPose(E, src, dst, self.camera.intrinsic)

    print(R, R.shape)
    print(t, t.shape)

  def triangulate(self, reference_frame):
    matches = self.get_matches(reference_frame)

  def update_pose(self, pose):
    self.pose = pose

  def annotate(self, reference_frame):
    img = self.feature.draw_keypoints((255, 0, 0))
    img = reference_frame.feature.draw_keypoints((0, 255, 0), img)

    for p1, p2 in self.matches:
      cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255))

    return img
