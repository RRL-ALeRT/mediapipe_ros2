#!/usr/bin/env python3

import copy
import csv
import itertools
import os
import sys
import cv2
from cv_bridge import CvBridge
import mediapipe as mp

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from loguru import logger

from ament_index_python import get_package_share_directory
keypoint_classifier_dir = os.path.join(get_package_share_directory('mediapipe_ros2'), 'keypoint_classifier')
sys.path.append(keypoint_classifier_dir)
from keypoint_classifier import KeyPointClassifier


class Mediapipe(Node):
    def __init__(self) -> None:
        # ROS2 init
        super().__init__('mp_hands')
        
        self.bridge = CvBridge()
        self.create_subscription(Image,"image_raw",self.imageflow_callback, 10)
        self.pub_gesture = self.create_publisher(String, '/recognized_gesture', 10)
        self.gesture = String()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.keypoint_classifier = KeyPointClassifier(model_path=keypoint_classifier_dir + '/keypoint_classifier.tflite')
        with open(keypoint_classifier_dir + '/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]

    def imageflow_callback(self,msg:Image) -> None:
        img_bgr = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        img_flipped = cv2.flip(img_bgr, 1)
        img_rgb = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        annotated_image = cv2.flip(img_bgr.copy(), 1)
        if results.multi_hand_landmarks is not None and 'Left' not in str(results.multi_handedness):
            for hand_landmarks in results.multi_hand_landmarks:
                
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                landmark_list = self.calc_landmark_list(annotated_image, hand_landmarks)

                pre_processed_landmark_list = self.pre_process_landmark(
                    landmark_list)
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                self.gesture.data = str(self.keypoint_classifier_labels[hand_sign_id])
                self.pub_gesture.publish(self.gesture)

        cv2.imshow("YOLOX",annotated_image)
        cv2.waitKey(1)
        
    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Key Point
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point


def main(args = None):
    rclpy.init(args=args)
    mp_class = Mediapipe()

    try:
        rclpy.spin(mp_class)
    except KeyboardInterrupt:
        pass
    finally:
        mp_class.destroy_node()
        rclpy.shutdown()
    
if __name__ == "__main__":
    main()
