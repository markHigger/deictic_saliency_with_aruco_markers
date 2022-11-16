import cv2
import numpy as np


class Object:

    def __init__(self, name, aruco_id, marker_length):
        self.name = name
        self.aruco_id = aruco_id
        self.marker_length = marker_length  # real size of marker in centimeters
        self.rect = None  # position boundries of aruco tag in pixel space

        #  Probably combine rvec and tvec into a single honogoneous transform
        self.rvec = None  # pose rvec
        self.tvec = None  # pose tvec

    def set_pose(self, position, K):
        self.rect = position
        rvec_marker, tvec_marker, _ = \
            cv2.aruco.estimatePoseSingleMarkers(position, self.marker_length, K, None)
        self.rvec = rvec_marker
        self.tvec = tvec_marker


class Parameters:
    """
    Used to find the permaters associated with the data and camera
    """

    def __init__(self):
        self.K = None  # Calibration matrix for camera
        self.marker_length = 3.5  # cm
        self.hand = Object('Hand', 876, self.marker_length)
        self.arm = Object('Arm', 965, self.marker_length)
        self.object_array = []  # List of objects in the scene
        self.object_array.append(Object('Rubix', 236, self.marker_length))
        self.object_array.append(Object('Cards', 358, self.marker_length))
        self.object_array.append(Object('Skull', 29, self.marker_length))
        self.object_array.append(Object('Coup', 15, self.marker_length))

    def calculate_k(self, im_width, im_height, fov):

        f_pixels = int((im_width / 2) / np.tan(np.deg2rad(fov)))
        c_x = int(im_width / 2)
        c_y = int(im_height / 2)
        self.K = np.matrix([[f_pixels, 0, c_x], [0, f_pixels, c_y], [0, 0, 1]]).astype(float)
        return self.K


def aruco_pose(image_frame, params):
    """
    Detects all aruco markers and poses in an image
    :param image_frame: still bgr image
    :param params: Parameter class object with information about the image and scene
                changes in params:
                    poses updated for the objects
    :return: list of poses and ids for each aruco tag detected
    """
    # ensure camera calibration matrix has been set
    try:
        assert (params.K is not None)
    except AssertionError as e:
        raise (AssertionError(
            "ERROR: K is not initialized in params please initialize camera calibration matrix with 'Perameters.calculate_k()'"
               % e))

    #  Use standard Aruco library
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    # Detect aruco markers in image
    markers_pos, markers_id, _ = cv2.aruco.detectMarkers(image_frame, aruco_dict)

    # Set Poses for each marker found
    for marker_idx, marker_id in enumerate(markers_id):
        marker_id = marker_id[0]
        marker_pos = markers_pos[marker_idx]

        # Set arm pose if marker is found
        if marker_id == params.arm.aruco_id:
            params.arm.set_pose(marker_pos, params.K)

        # Set hand pose if marker is found
        elif marker_id == params.hand.aruco_id:
            params.hand.set_pose(marker_pos, params.K)

        # Loop through scene objects and set pose for each object detected
        else:
            for scene_object in params.object_array:
                if marker_id == scene_object.aruco_id:
                    scene_object.set_pose(marker_pos, params.K)

    # DEBUG: Show images for debug
    img_debug = image_frame.copy()
    cv2.aruco.drawDetectedMarkers(img_debug, markers_pos)
    cv2.imshow('DEBUG: ARUCO', img_debug)
    cv2.waitKey(0)


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

