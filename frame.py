import cv2


class Frame:
  def __init__(self, idx, pose, feature, cam):
    self.idx = idx
    self.pose = pose # g2o.Isometry3d
    self.feature = feature 
    self.cam = cam
    self.image = feature.image

    self.transform_matrix = pose.inverse().matrix()[:3] # shape (3, 4)
    self.projection_matrix = self.cam.intrinsic.dot(self.transform_matrix) # world frame to image
  
  # world coordinates -> camera frame
  def transform(self, points):
    R = self.transform_matrix[:3, :3]
    if points.ndim == 1:
      t = self.transform_matrix[:3, 3]
    else:
      t = self.transform_matrix[:3, 3:]
    return np.dot(R, points) + t

  # camera frame -> image coordinates
  def project(self, points):
    projection = self.cam.intrinsic.dot(points /points[-1:])
    return projection[:2], points[-1]

  def find_matches(self, points, descriptors):
    points = np.array(points).T
    proj, _ = self.project(self.transform(points))
    proj = proj.transpose()
    return self.feature.find_matches(proj, descriptors)


