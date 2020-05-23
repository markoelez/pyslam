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
- For video files `chmod +x slam.py && ./slam.py <test.mp4>`
- For Kitti dataset `chmod +x pyslam.py && ./pyslam.py </path/to/kitti/sequences/00>`

## License 
All of this code is MIT licensed. Videos and libraries follow their respective licenses.
