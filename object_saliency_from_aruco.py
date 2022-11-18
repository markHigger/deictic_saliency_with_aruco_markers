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
        self.R_aruco2Object = None #manually defined
        self.H_aruco2Object = None
        self.R_object2world = None
        self.H_object2world = None

    def set_pose(self, position, params):

        # Marker boundry positions
        self.rect = position

        # find rodregues and translation vectors of a marker
        rvec_marker, tvec_marker, _ = \
            cv2.aruco.estimatePoseSingleMarkers(position, self.marker_length, params.K, None)
        rvec_marker = rvec_marker[0][0]
        tvec_marker = tvec_marker[0][0]
        self.rvec = rvec_marker
        self.tvec = tvec_marker

        # convert to homogounous matrix
        R_aruco2world, _ = cv2.Rodrigues(rvec_marker)
        t = np.matrix(tvec_marker).T
        H_aruco2world = params.make_homogenous_transform(R_aruco2world, t)
        self.H_object2world = H_aruco2world @ self.H_aruco2Object
        self.R_object2world = self.H_object2world[0:4, 0:4]


class SceneParameters:
    """
    Used to find the permaters associated with the data and camera
    """

    def __init__(self):
        self.K = None  # Calibration matrix for camera
        # self.marker_length = 3.5  # cm
        self.marker_length = 17
        self.hand = SceneObject('Hand', 876, self.marker_length)
        self.hand.H_aruco2Object = np.matrix([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]])
        self.arm = SceneObject('Arm', 965, self.marker_length)
        self.arm.H_aruco2Object = np.matrix([[-1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]])
        # self.arm = SceneObject('Arm', 756, self.marker_length)
        self.game_list = []  # List of objects in the scene
        self.game_list.append(SceneObject('Rubix', 236, self.marker_length))
        self.game_list[-1].H_aruco2Object = np.matrix([[1, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]])
        self.game_list.append(SceneObject('Cards', 358, self.marker_length))
        self.game_list[-1].H_aruco2Object = np.matrix([[1, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]])
        self.game_list.append(SceneObject('Skull', 29, self.marker_length))
        self.game_list[-1].H_aruco2Object = np.matrix([[1, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]])
        self.game_list.append(SceneObject('Coup', 15, self.marker_length))
        self.game_list[-1].H_aruco2Object = np.matrix([[1, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]])
        # self.gesture_unit_vector = None
        # self.gesture_unit_vector_2d = None
        self.gesture_origin_3d = None
        self.gesture_origin_pixels = None
        self.gesture_R = None
        self.gesture_H = None
        self.point_line = None

    def calculate_k(self, im_width, im_height, fov):
        """
        Creates camera calibration Matrix from input permatmers
        :param im_width: image width (pixels)
        :param im_height: image height (pixels)
        :param fov: Camera field of view in degrees
        :return: K: calibration parameter (also stored in self)

        """
        f_pixels = int((im_height/ 2) / np.tan(np.deg2rad(fov)))
        c_x = int(im_width / 2)
        c_y = int(im_height / 2)
        self.K = np.matrix([[f_pixels, 0, c_x], [0, f_pixels, c_y], [0, 0, 1]]).astype(float)
        return self.K

    def set_gesture_origin(self, origin_3d, H_gesture2world):
        self.gesture_origin_3d = origin_3d
        self.gesture_origin_pixels = self.homgenous_3d_to_camera_pixels(origin_3d)
        self.gesture_R = H_gesture2world[0:4][0:4]
        self.gesture_H = H_gesture2world
        pointing_vector_bf = np.matrix([1000, 0, 0]).T
        self.point_line = self.gesture_H @ self.make_homogenous(pointing_vector_bf)
        self.point_line = self.homgenous_3d_to_camera_pixels(self.point_line)
        pass

    def make_homogenous(self, vector):
        """
        turns a 3x1 vector into a homogenous vector
        :param vector:
        :return:
        """
        return np.concatenate([vector, np.matrix([[1]])], axis=0)

    def make_homogenous_transform(self, R, t):
        """
        Makes an H matrix from rotation and translation
        :param R: 3x3 np matrix for rotation from A to B
        :param t: 3x1 np matrix for translation in frame A
        :return: 4x4 np matrix for homogenous transform
        """
        H = np.concatenate([R, t], 1)
        H = np.concatenate([H, np.matrix([0, 0, 0, 1])], 0)
        return H

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
            params.arm.set_pose(marker_pos, params)

        # Set hand pose if marker is found
        elif marker_id == params.hand.aruco_id:
            params.hand.set_pose(marker_pos, params)

        # Loop through scene objects and set pose for each object detected
        else:
            for game in params.game_list:
                if marker_id == game.aruco_id:
                    game.set_pose(marker_pos, params)




def calculate_arm_pose(params):
    """
    :param params:Parameter class object with information about the image and scene
    :return: unit vector and 3d origin point of arm
    """
    # TODO: calibrate arm and hand pose

    # generate unit vector
    unit_vector_x_bf = np.matrix([-1, 0, 0]).T
    origin_body_frame = np.matrix([0, 0, 0, 1]).T

    if params.arm.R_object2world is not None and params.hand.R_object2world is not None:
        # TODO handle case where hand is present
        print('case where hand is present is not defined yet')

    elif params.hand.R_object2world is not None:
        # TODO handle case where hand is present
        print('case where hand is present is not defined yet')

    elif params.arm.R_object2world is not None:
        # Find arm unit vector
        # arm_vector_x = params.arm.R @ unit_vector_x_bf
        # arm_vector_x = arm_vector_x / np.linalg.norm(arm_vector_x)

        # Find arm position in real 3d space
        arm_origin = params.arm.H_object2world @ origin_body_frame
        params.set_gesture_origin(arm_origin, params.arm.H_object2world)
    else:
        print('No gesture is detected')


def calculate_object_vectors(params):
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

