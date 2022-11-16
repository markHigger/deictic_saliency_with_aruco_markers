import cv2
import numpy as np
from object_saliency_from_aruco import *


def single_image_test(image_path):
    img = cv2.imread(image_path)
    param = Parameters()
    param.calculate_k(img.shape[1], img.shape[0], 60)
    aruco_poses = aruco_pose_test(img, param)
    pass


def aruco_pose_test(image, params):
    poses = aruco_pose(image, params)
    pass


def calculate_arm_pose_test():
    pass


def calculate_object_vectors_test():
    pass


if __name__ == "__main__":
    image_path = "test_data/IMG_1070.jpg"
    single_image_test(image_path)
    pass

