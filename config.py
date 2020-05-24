import cv2

class KittiConfig:
  def __init__(self):
    self.feature_detector = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=1,
        edgeThreshold=31)

    self.descriptor_extractor = self.feature_detector

    self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    self.matching_distance = 30

    # img params
    self.view_image_width = 400
    self.view_image_height = 130
    self.view_camera_width = 0.75
    self.view_viewpoint_x = 0
    self.view_viewpoint_y = -500   # -10
    self.view_viewpoint_z = -100   # -0.1
    self.view_viewpoint_f = 2000

    self.image_width = 1241
    self.image_height= 376
