#!/Users/marko/anaconda3/envs/dl/bin/python3
import sys
import cv2
import pygame
import numpy as np
from pygame.locals import DOUBLEBUF
from display import Display2D
from frame import Frame

sys.path.append('lib/')

class SLAM:
    def __init__(self, W, H, K):
        self.W, self.H = W, H
        # camera matrix
        self.K = K

    def process_frame(self, img):
        frame = Frame(img, self.K)
        img = frame.annotate(img)

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

    # camera matrix
    F = 525
    # adjust img origin (bottom-left) to match camera origin (middle)
    px, py = W//2, H//2
    K = np.array([[F, 0, px], [0, F, py], [0, 0, 1]]) 
    
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
