#!/Users/marko/anaconda3/envs/dl/bin/python3
import time
import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform 


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

    # filter
    ret = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]
            if m.distance < 32:
                ret.append((p1, p2))
   
    assert len(ret) >= 8

    ret = np.array(ret)
    
    src = ret[:, 0]
    dst = ret[:, 1]

    model, inliers = ransac((src, dst),
                             FundamentalMatrixTransform,
                             min_samples = 8,
                             residual_threshold=1, 
                             max_trials=100)

    ret = ret[inliers]

    return ret 
    

class Frame:
    def __init__(self, img, K):
        self.K = np.array(K)
        if img is not None:
            self.h, self.w = img.shape[0:2]
            self.kps, self.des = get_features(img)
            self.img = img
    
    def show(self, img, f2, matches=None):
        if matches is None:
            return img
        print('matches %d' % len(matches))
        for p1, p2 in matches:
            cv2.circle(img, (p1[0], p1[1]), color=(0, 255, 0), radius=2)
            cv2.circle(img, (p2[0], p2[1]), color=(0, 255, 255), radius=1)
            img = cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), thickness=1)

        return img
    
