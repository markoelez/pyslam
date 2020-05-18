#!/Users/marko/anaconda3/envs/dl/bin/python3
import sys
import cv2
import pygame
import numpy as np
from pygame.locals import DOUBLEBUF
from display import Display2D
from frame import Frame, get_matches
from mapp import Map 
from helpers import FtoRt

sys.path.append('lib/')

class SLAM:
    def __init__(self, W, H, K):
        self.W, self.H = W, H
        # map
        self.map = Map()
        # camera matrix
        self.K = K

    def process_frame(self, img):
        frame = Frame(img, self.K)

        # add frame to map
        self.map.add_frame(frame) 
        if len(self.map.frames) > 2:
            f1 = self.map.frames[-1]
            f2 = self.map.frames[-2]

            matches, F = get_matches(f1, f2)

            R, t = FtoRt(F, self.K)

            img = frame.show(img, f2, matches)
        else:
            img = frame.show(img, None, None)

        print(img.shape)
        return img 

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('%s <video.mp4>' % sys.argv[0])
        exit(-1)

    cap = cv2.VideoCapture(sys.argv[1])

    # frame props
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//2

    # focal length
    F = 1 

    # camera matrix
    # adjust mapping: img coords have origin at center, digital have origin bottom-left
    K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]]) 
    
    slam = SLAM(W, H, K)
    disp = Display2D(W, H)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (W, H))

        if ret == True:
            img = slam.process_frame(frame)
            disp.paint(img)
        else:
            break
