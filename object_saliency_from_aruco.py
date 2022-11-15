import cv2
import numpy as np


def aruco_pose(image_frame):
    """
    Detects all aruco markers and poses in an image
    :param image_frame: still bgr image
    :return: list of poses and ids for each aruco tag detected
    """

    pose_vector = []
    pass


def calculate_arm_pose(arm_and_hand_poses):
    """
    :param arm_and_hand_poses: list of poses of the arm and hand
    :return: unit vector and 3d origin point of arm
    """
    pass


def calculate_object_vectors(object_poses, arm_position):
    """
    :param object_poses: list of the pose for each object
    :param arm_position: position of the arm
    :return: list of unit vectors for each object
    """
    pass


def find_salient_object(object_vectors, arm_direction):
    """
    :param object_vectors: list of unit vectors for each object
    :param arm_direction: unit vector of arm pointing
    :return: index of object that is being pointed to
    """
    pass


def highlight_object(image, aruco_id):
    """
    :param image: bgr image frame to draw on
    :param aruco_id: id of object to highlight
    :return:
    """
    pass

