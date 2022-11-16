import cv2
import numpy as np
from object_saliency_from_aruco import *


def single_image_test(image_path):
    img = cv2.imread(image_path)
    param = Parameters()
    param.calculate_k(img.shape[1], img.shape[0], 30)
    aruco_pose_test(img, param)
    cv2.waitKey(0)
    pass


def aruco_pose_test(image, params):
    aruco_pose(image, params)
    # DEBUG: Show images for debug
    img_debug = image.copy()

    if params.arm.rvec is not None:
        cv2.aruco.drawAxis(img_debug, params.K, None, params.arm.rvec, params.arm.tvec, 3)

    if params.hand.rvec is not None:
        cv2.aruco.drawAxis(img_debug, params.K, None, params.hand.rvec, params.hand.tvec, 3)

    for object_in_scene in params.object_array:
        if object_in_scene.rvec is not None:
            cv2.aruco.drawAxis(img_debug, params.K, None, object_in_scene.rvec, object_in_scene.tvec, 3)

    cv2.imshow('DEBUG: ARUCO Axes', img_debug)

    pass


def calculate_arm_pose_test():
    pass


def calculate_object_vectors_test():
    pass


if __name__ == "__main__":
    image_path = "test_data/IMG_1048.jpg"
    single_image_test(image_path)
    pass

