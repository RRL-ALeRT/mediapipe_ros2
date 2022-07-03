import launch
import launch_ros.actions
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    pose = launch_ros.actions.Node(
        package="mediapipe_ros2", executable="pose",
    )

    video_device = LaunchConfiguration('video_device', default='/dev/video0')
    webcam = launch_ros.actions.Node(
        package="v4l2_camera", executable="v4l2_camera_node",
        parameters=[
            {"image_size": [640,480]},
            {"video_device": video_device},
        ],
    )

    return launch.LaunchDescription([
        pose,
        webcam,
    ])
