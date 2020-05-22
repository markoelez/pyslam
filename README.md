# pyslam

This project demonstrates a simple implementation of a monocular SLAM system using ORB feature detectors

## Features
- MP4 video loader
- KITTI Odometry dataset loader
- Point cloud visualizer

## Requirements
- Python 3.7+
- numpy (for matrix manipulation)
- OpenCV (for image processing)
- g2o (for pose estimatation)
- pangolin (for visualization)

## Usage
for video files `chmod +x slam.py && ./slam.py <test.mp4>`
for Kitti dataset `chmod +x pyslam.py && ./pyslam.py </path/to/kitti/sequences/00>`

