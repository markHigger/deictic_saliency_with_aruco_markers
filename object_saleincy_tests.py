import cv2
import numpy as np
from object_saliency_from_aruco import *


# def fov_test(image_path):
#     img = cv2.imread(image_path)
#     param = SceneParameters()
#     param.calculate_k(img.shape[1], img.shape[0], 80)
#     aruco_pose_test(img, param)
#     pass


def single_image_test(image_path):
    """
    Runs through a single image and displays parameters
    :param image_path: string - image file name
    :return:
    """
    # read image
    img = cv2.imread(image_path)

    # Run base parameters with scene
    param = SceneParameters()

    # Find camera calibration perameters
    param.calculate_k(img.shape[1], img.shape[0], 45)

    # Find aurco pose of each object in scene
    aruco_pose_test(img, param)

    # Calculate gesture origin and direction
    calculate_arm_pose_test(img, param)

    # Find which object is being pointed at
    find_salient_object_test(img, param)

    cv2.waitKey(0)
    pass


def aruco_pose_test(image, params):
    """
    Find pose of each aruco marker and object and draws the axes of each marker
    :param image: Image of scene for aruco pose calculation and image display
    :param params: SceneParameter Object associated with current scene
    :return:
    """
    # Find poses of each object in scene
    aruco_pose(image, params)

    # Show images for debug
    img_debug = image.copy()

    # Draw axes for each aruco tag that has been found
    if params.arm.rvec is not None:
        cv2.aruco.drawAxis(img_debug, params.K, None, params.arm.rvec, params.arm.tvec, 20)

    if params.hand.rvec is not None:
        cv2.aruco.drawAxis(img_debug, params.K, None, params.hand.rvec, params.hand.tvec, 20)

    for game in params.game_list:
        if game.rvec is not None:
            cv2.aruco.drawAxis(img_debug, params.K, None, game.rvec, game.tvec, 20)

    cv2.imshow('DEBUG: ARUCO Axes', img_debug)

    pass


def calculate_arm_pose_test(img, params):
    """
    Find gesture pose and draw a line to denote where the gesture is pointing
    :param img: cv image to draw on
    :param param: SceneParameter Object associated with current scene (with aruco poses calculated)
    :return:
    """

    # Find gesture pose
    calculate_arm_pose(params)

    # Plot gesture pose
    p1 = params.gesture_origin_pixels
    p2 = params.point_line

    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    cv2.line(img, p1, p2, (0, 0, 255), 20)
    cv2.imshow('Pointing Vector', img)
    pass


def find_salient_object_test(img, param):
    """
    Finds and displays object closest to the gesture
    :param img: Image to display gesture
    :param param:Scene Parameter object
    :return:
    """
    # Calculate salient object
    winner = find_salient_object(param)

    # Draw lines to each object from origin of gesture
    forward_vector_bf = np.matrix([1, 0, 0]).T * 200
    for game_idx, game in enumerate(param.game_list):
        if game.R_gesture2Object is None:
            continue
        # Calculate gesture to object vectors
        R_op2gesture = np.linalg.inv(game.R_gesture2Object)
        forward_vector_gf = R_op2gesture @ forward_vector_bf
        vector_gf = param.gesture_H @ param.make_homogenous(forward_vector_gf)
        vector_pixels = param.homgenous_3d_to_camera_pixels(vector_gf)

        p1 = param.gesture_origin_pixels
        p2 = vector_pixels
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))

        cv2.line(img, p1, p2, (0, int(255/(game_idx+1)), 0), 20)
        cv2.imshow('Pointing Vector', img)
        print(game.name)
    print('Closest Game: ', winner.name)
    pass


def calculate_object_vectors_test():
    pass


if __name__ == "__main__":
    # image_path = "test_data/IMG_1069.jpg"
    # image_path = "test_data/IMG_1048.jpg"  # Broken (No Arm)
    # image_path = "test_data/IMG_1066.jpg"
    image_path = "test_data/IMG_1056.jpg"
    # image_path = "test_data/IMG_1070.jpg"  # Incorrect?
    # image_path = "test_data/IMG_1052.jpg"
    # image_path = "test_data/IMG_1067.jpg"
    # image_path = "test_data/single_marker.jpg"
    # fov_test(image_path)
    single_image_test(image_path)
    pass

