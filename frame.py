import cv2
import numpy as np


def extractFeatures(img):
    orb = cv2.ORB_create()

    # detection
    features = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7) 

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]  
    kps, des = orb.compute(img, kps)  

    return np.array([(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]), des

class Frame:
    def __init__(self, img, K):
        self.K = np.array(K)
        if img is not None:
            self.h, self.w = img.shape[0:2]
    
    def annotate(self, img):
        kps, des = extractFeatures(img)
        for (u, v) in kps:
            cv2.circle(img, (u, v), color=(0, 255, 0), radius=1)
        return img

