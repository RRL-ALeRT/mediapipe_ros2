# Mediapipe ROS2 Wrapper
## Installation
```
sudo apt install ros-foxy-v4l2-camera -y
pip3 install mediapipe tensorflow sklearn
```
## Launch
### Hand Gesture
```
ros2 launch mediapipe_ros2 hand_gesture.launch.py
```
### Pose Gesture
```
ros2 launch mediapipe_ros2 pose_gesture.launch.py
```
## See output in ROS2 Topic using
```
ros2 topic echo /recognized_gesture
```
