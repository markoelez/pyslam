import time
import cv2
import numpy as np


def get_features(img):
    orb = cv2.ORB_create()

    # detection
    features = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7) 

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]  
    kps, des = orb.compute(img, kps)  

    return np.array([(int(kp.pt[0]), int(kp.pt[1])) for kp in kps]), des

def get_matches(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    return matches
    

class Frame:
    def __init__(self, img, K):
        self.K = np.array(K)
        if img is not None:
            self.h, self.w = img.shape[0:2]
            self.kps, self.des = get_features(img)
            self.img = img
    
    def show(self, img, f2, matches=None):
        for (u, v) in self.kps:
            cv2.circle(img, (u, v), color=(0, 255, 0), radius=2)

        if matches is not None:
            for m, n in matches:
                # simplified Lowe's ratio test
                if m.distance < 0.75*n.distance:
                    p1 = self.kps[m.queryIdx]
                    p2 = f2.kps[m.trainIdx]

                    if m.distance < 32:
                        print(p1, p2)
                        
                        (x1, y1) = p1
                        (x2, y2) = p2

                        img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)

        return img
    
