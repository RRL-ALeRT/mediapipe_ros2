import launch
import launch_ros.actions
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    hands = launch_ros.actions.Node(
        package="mediapipe_ros2", executable="hands",
    )

    video_device = LaunchConfiguration('video_device', default='/dev/video0')
    webcam = launch_ros.actions.Node(
        package="v4l2_camera", executable="v4l2_camera_node",
        parameters=[
            {"image_size": [640,480]},
            {"video_device": video_device},
        ],
    )
    gesture_to_cmdvel = launch_ros.actions.Node(
        package="mediapipe_ros2", executable="gesture_to_cmdvel"
    )

    return launch.LaunchDescription([
        hands,
        webcam,
        gesture_to_cmdvel,
    ])
