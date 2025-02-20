import cv2
import numpy as np
from object_saliency_from_aruco import *


# def fov_test(image_path):
#     img = cv2.imread(image_path)
#     param = SceneParameters()
#     param.calculate_k(img.shape[1], img.shape[0], 80)
#     aruco_pose_test(img, param)
#     pass


def video_test(video_path):

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")



    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    # Run base parameters with scene
    scene_video = SceneParameters()
    scene_video.calculate_k(frame_width, frame_height, 40)
    # scene_video.calculate_k(frame_width, frame_height, 20)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Blank frame', frame)

        # Find aurco pose of each object in scene
        aruco_pose_test(frame, scene_video)

        calculate_arm_pose_test(frame, scene_video)

        # Find which object is being pointed at
        find_salient_object_test(frame, scene_video)

        highlight_salient_object(frame, scene_video)

        cv2.waitKey(1)

    pass

def single_image_test(image_path):
    """
    Runs through a single image and displays parameters
    :param image_path: string - image file name
    :return:
    """
    print(image_path)
    # read image
    img = cv2.imread(image_path)

    # Run base parameters with scene
    param = SceneParameters()

    # Find camera calibration perameters
    param.calculate_k(img.shape[1], img.shape[0], 40)

    # Find aurco pose of each object in scene
    aruco_pose_test(img, param)

    # Calculate gesture origin and direction
    calculate_arm_pose_test(img, param)

    # Find which object is being pointed at
    find_salient_object_test(img, param)

    highlight_salient_object(img, param)

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
    if params.arm.is_found:
        cv2.aruco.drawAxis(img_debug, params.K, None, params.arm.rvec, params.arm.tvec, 3)

    if params.hand.is_found:
        cv2.aruco.drawAxis(img_debug, params.K, None, params.hand.rvec, params.hand.tvec, 3)

    for game in params.game_list:
        if game.is_found:
            cv2.aruco.drawAxis(img_debug, params.K, None, game.rvec, game.tvec, 3)

    cv2.imshow('DEBUG: ARUCO Axes', img_debug)

    pass


def calculate_arm_pose_test(img, params):
    """
    Find gesture pose and draw a line to denote where the gesture is pointing
    :param img: cv image to draw on
    :param param: SceneParameter Object associated with current scene (with aruco poses calculated)
    :return:
    """

    if params.arm.is_found:
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
    winner, r_min = find_salient_object(param)

    if winner is None:
        return 0

    img_objects = img.copy()
    # Draw lines to each object from origin of gesture
    forward_vector_bf = np.matrix([1, 0, 0]).T * 33
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

        cv2.line(img_objects, p1, p2, (0, int(255/(game_idx+1)), 0), 20)
        cv2.imshow('Pointing Vector', img_objects)
    if param.salient_game is not None:
        print('Closest Game: ', param.salient_game.name)

    cv2.imshow('Pointing Vector', img_objects)



def highlight_salient_object(img, scene_params):

    if scene_params.arm.is_found:
        # Plot gesture pose
        p1 = scene_params.gesture_origin_pixels
        p2 = scene_params.point_line

        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        cv2.line(img, p1, p2, (0, 0, 255), 20)

    if scene_params.salient_game is not None:
        scene_params.salient_game.highlight_object(img, scene_params.K)
    cv2.imshow('Deictic Saliency', img)

if __name__ == "__main__":

    # Wrong Direction
    # single_image_test('test_data/IMG_1045.jpg')
    # single_image_test('test_data/IMG_1046.jpg')
    # single_image_test('test_data/IMG_1047.jpg')
    # image_path = "testq_data/IMG_1048.jpg"  # Broken (No Arm)
    # single_image_test(image_path)
    # single_image_test('test_data/IMG_1049.jpg')
    # single_image_test('test_data/IMG_1050.jpg')
    # single_image_test('test_data/IMG_1051.jpg')
    # image_path = "test_data/IMG_1052.jpg"
    # single_image_test(image_path)
    # single_image_test('test_data/IMG_1053.jpg')
    # single_image_test('test_data/IMG_1054.jpg')
    # single_image_test('test_data/IMG_1055.jpg')
    image_path = "test_data/IMG_1056.jpg"
    single_image_test('test_data/IMG_1065.jpg')
    single_image_test(image_path)
    image_path = "test_data/IMG_1066.jpg"
    single_image_test(image_path)
    image_path = "test_data/IMG_1067.jpg"
    single_image_test(image_path)
    single_image_test('test_data/IMG_1068.jpg')
    image_path = "test_data/IMG_1069.jpg"
    # single_image_test(image_path)
    # image_path = "test_data/IMG_1070.jpg"  # Incorrect?
    # single_image_test(image_path)


    # video_test('test_data/IMG_1063.MOV')
    # video_test('test_data/IMG_1058.MOV')
    # video_test('test_data/IMG_1064.MOV')
    # video_test('test_data/IMG_1059.MOV')
    # video_test(0)

    pass

