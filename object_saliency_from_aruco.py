import cv2
import numpy as np


class SceneObject:

    def __init__(self, name, aruco_id, marker_length):
        self.name = name
        self.aruco_id = aruco_id
        self.marker_length = marker_length  # real size of marker in centimeters
        self.rect = None  # position boundries of aruco tag in pixel space

        #  rvec and tvec can be used directly for aruco functions
        self.rvec = None  # pose rvec
        self.tvec = None  # pose tvec

        # R is ratoation matrix used for arm and hand unit vectors
        # H is homogenous transformation matrix used for coordinate transforms
        self.R = None
        self.H = None

    def set_pose(self, position, K):

        # Marker boundry positions
        self.rect = position

        # find rodregues and translation vectors of a marker
        rvec_marker, tvec_marker, _ = \
            cv2.aruco.estimatePoseSingleMarkers(position, self.marker_length, K, None)
        rvec_marker = rvec_marker[0][0]
        tvec_marker = tvec_marker[0][0]
        self.rvec = rvec_marker
        self.tvec = tvec_marker

        # convert to homogounous matrix
        R, _ = cv2.Rodrigues(rvec_marker)
        t = np.matrix(tvec_marker).T
        H = np.concatenate([R, t], 1)
        H = np.concatenate([H, np.matrix([0, 0, 0, 1])], 0)
        self.R = R
        self.H = H


class SceneParameters:
    """
    Used to find the permaters associated with the data and camera
    """

    def __init__(self):
        self.K = None  # Calibration matrix for camera
        self.marker_length = 3.5  # cm
        self.hand = SceneObject('Hand', 876, self.marker_length)
        self.arm = SceneObject('Arm', 965, self.marker_length)
        self.game_list = []  # List of objects in the scene
        self.game_list.append(SceneObject('Rubix', 236, self.marker_length))
        self.game_list.append(SceneObject('Cards', 358, self.marker_length))
        self.game_list.append(SceneObject('Skull', 29, self.marker_length))
        self.game_list.append(SceneObject('Coup', 15, self.marker_length))
        self.gesture_unit_vector = None
        self.gesture_unit_vector_2d = None
        self.gesture_origin_3d = None
        self.gesture_origin_pixels = None

    def calculate_k(self, im_width, im_height, fov):
        """
        Creates camera calibration Matrix from input permatmers
        :param im_width: image width (pixels)
        :param im_height: image height (pixels)
        :param fov: Camera field of view in degrees
        :return: K: calibration parameter (also stored in self)

        """
        f_pixels = int((im_width / 2) / np.tan(np.deg2rad(fov)))
        c_x = int(im_width / 2)
        c_y = int(im_height / 2)
        self.K = np.matrix([[f_pixels, 0, c_x], [0, f_pixels, c_y], [0, 0, 1]]).astype(float)
        return self.K

    def set_gesture_origin(self, unit_vector, origin_3d):
        self.gesture_unit_vector = unit_vector
        unit_vector_homgenous = np.concatenate([unit_vector, np.matrix([[1]])], axis=0)
        unit_vector_pixels = self.homgenous_3d_to_camera_pixels(unit_vector_homgenous)
        unit_vector_2d = unit_vector_pixels / np.linalg.norm(unit_vector_pixels)
        self.gesture_unit_vector_2d = unit_vector_2d
        self.gesture_origin_3d = origin_3d
        self.gesture_origin_pixels = self.homgenous_3d_to_camera_pixels(origin_3d)

    def homgenous_3d_to_camera_pixels(self, vector_h3d):
        # Find arm position in camera frame
        mat_dim = np.matrix([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0]])
        vector_2d = self.K @ mat_dim @ vector_h3d
        vector_2d = vector_2d[0:2] / vector_2d[2]
        return vector_2d.astype(int)
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
    markers_pos, markers_id, rejected = cv2.aruco.detectMarkers(image_frame, aruco_dict)
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
            for game in params.game_list:
                if marker_id == game.aruco_id:
                    game.set_pose(marker_pos, params.K)




def calculate_arm_pose(params):
    """
    :param params:Parameter class object with information about the image and scene
    :return: unit vector and 3d origin point of arm
    """
    # TODO: calibrate arm and hand pose

    # generate unit vector
    unit_vector_body_frame = np.matrix([1, 1, 1]).T
    origin_body_frame = np.matrix([0, 0, 0, 1]).T

    if params.arm.R is not None and params.hand.R is not None:
        # TODO handle case where hand is present
        print('case where hand is present is not defined yet')

    elif params.hand.R is not None:
        # TODO handle case where hand is present
        print('case where hand is present is not defined yet')

    elif params.arm.R is not None:
        # Find arm unit vector
        arm_vector = params.arm.R @ unit_vector_body_frame
        arm_vector = arm_vector / np.linalg.norm(arm_vector)

        # Find arm position in real 3d space
        arm_origin = params.arm.H @ origin_body_frame
        params.set_gesture_origin(arm_vector, arm_origin)


    else:
        print('No gesture is detected')



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
    # Use cosine similarity to find most similar angles
    # Similarity = (A.B) / (| | A | |.| | B | |)
    pass


def highlight_object(image, aruco_id):
    """
    :param image: bgr image frame to draw on
    :param aruco_id: id of object to highlight
    :return:
    """
    pass

