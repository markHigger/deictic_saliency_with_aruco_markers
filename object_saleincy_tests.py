import cv2
import numpy as np
from object_saliency_from_aruco import *

def fov_test(image_path):
    img = cv2.imread(image_path)
    param = SceneParameters()
    param.calculate_k(img.shape[1], img.shape[0], 35)
    aruco_pose_test(img, param)
    pass


def single_image_test(image_path):
    img = cv2.imread(image_path)
    param = SceneParameters()
    param.calculate_k(img.shape[1], img.shape[0], 40)
    aruco_pose_test(img, param)
    calculate_arm_pose_test(img, param)
    cv2.waitKey(0)
    pass


def aruco_pose_test(image, params):
    aruco_pose(image, params)

    # Show images for debug
    img_debug = image.copy()

    if params.arm.rvec is not None:
        cv2.aruco.drawAxis(img_debug, params.K, None, params.arm.rvec, params.arm.tvec, 3)

    if params.hand.rvec is not None:
        cv2.aruco.drawAxis(img_debug, params.K, None, params.hand.rvec, params.hand.tvec, 3)

    for game in params.game_list:
        if game.rvec is not None:
            cv2.aruco.drawAxis(img_debug, params.K, None, game.rvec, game.tvec, 3)

    cv2.imshow('DEBUG: ARUCO Axes', img_debug)

    pass


def calculate_arm_pose_test(img, param):
    calculate_arm_pose(param)
    p1 = param.gesture_origin_pixels
    p2 = param.point_line

    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    cv2.line(img, p1, p2, (0, 0, 255), 20)
    cv2.imshow('Pointing Vector', img)
    pass


def calculate_object_vectors_test():
    pass


if __name__ == "__main__":
    image_path = "test_data/IMG_1052.jpg"
    # image_path = "test_data/single_marker.jpg"
    # fov_test(image_path)
    single_image_test(image_path)
    pass

