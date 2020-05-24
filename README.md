# pyslam

This project demonstrates a simple implementation of a monocular SLAM system using ORB feature detectors

## Features
- MP4 video loader
- KITTI Odometry dataset loader
- Point cloud visualizer

## Requirements
- Python 3.7+
- numpy
- OpenCV
- g2o
- pangolin

For g2o and pangolin python bindings, use the forks in my github.

## Usage
- For video files `chmod +x system.py && ./system.py --type <VIDEO> --path <test.mp4>`
- For Kitti dataset `chmod +x system.py && ./system.py --type <KITTI> --path </path/to/kitti/dataset/sequences/00>`

## License 
All of this code is MIT licensed. Videos and libraries follow their respective licenses.
