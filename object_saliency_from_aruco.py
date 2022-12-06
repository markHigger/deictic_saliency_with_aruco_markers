import cv2
import numpy as np
import scipy.spatial.transform as sci_trans


class SceneObject:

    def __init__(self, name, aruco_id, marker_length):
        # Sets the initialization peramters the object
        self.name = name
        self.aruco_id = aruco_id
        self.marker_length = marker_length  # real size of marker in centimeters

        '''Must be manually defined outside self'''
        self.H_aruco2Object = None

        ''' Filled out in self.set_pose'''
        # bounding box (in pixel-space) of the aruco tag associated with object
        self.rect = None  # position boundries of aruco tag in pixel space
        #  rvec and tvec can be used directly for aruco functions
        self.rvec = None  # pose rvec
        self.tvec = None  # pose tvec

        # R is ratoation matrix used for arm and hand unit vectors
        # H is homogenous transformation matrix used for coordinate transforms
        self.R_object2world = None
        self.H_object2world = None

        ''' calculated in self.calculate_gesture_to_object_rotation'''
        # Rotation between gesture direction and the direction from gesture to object
        self.R_gesture2Object = None

        # Check if object is found in this frame
        self.is_found = 0


    def set_pose(self, position, params):
        """
        Sets the pose and related object perameterers of an object in the scene from aruco tags
        :param position: list of 4 2d pixel poses (ints) of the bouding boxes of the aruco marker
        :param params: Scene Parameters object used for camera peramters and supporting functions
        :return:
        """

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
        self.H_object2world = H_aruco2world @ np.linalg.inv(self.H_aruco2Object)
        self.R_object2world = self.H_object2world[0:3, 0:3]

    def calculate_gesture_to_object_rotation(self, gesture_origin, gesture_unit_vector):
        """
        Calculates the rotation matrix between the gesture pointing direction and the direction of the gesture to a particular object
        :param gesture_origin: 4x1 np matrix of homogenous coordinates in real 3d space (camera frame)
        :param gesture_unit_vector: 3x1 np matrix of the normalized pointing direction of the gesture
        :return: R_gesture2Object the 3x3 np matrix of the rotation between the actual gesture and the direction of the gesture to object
        """
        # Find unit vector between a game and the gesture
        pos_diff = np.matrix(self.tvec).T - gesture_origin[0:3]
        object_unit_vector = pos_diff / np.linalg.norm(pos_diff)

        # Find the rotation matrix between unit vectors
        a = np.squeeze(np.asarray(object_unit_vector))
        b = np.squeeze(np.asarray(gesture_unit_vector))
        v = np.cross(a, b)
        c = np.dot(a, b.T)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        self.R_gesture2Object = R
        return R

    def highlight_object(self, img, K):
        corners = self.rect
        cv2.aruco.drawDetectedMarkers(img, [corners])
        draw_first_last_connected(img, corners[0])
        cv2.aruco.drawAxis(img, K, None, self.rvec, self.tvec, 3)

class SceneParameters:
    """
    Used to find the permaters associated with the data and camera
    """

    def __init__(self):
        self.K = None  # Calibration matrix for camera
        self.marker_length = 3.5  # cm
        # self.marker_length = 17
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

        self.salient_game = None

    def calculate_k(self, im_width, im_height, fov):
        """
        Creates camera calibration Matrix from input permatmers
        :param im_width: image width (pixels)
        :param im_height: image height (pixels)
        :param fov: Camera field of view in degrees
        :return: K: calibration parameter (also stored in self)

        """
        f_pixels = (im_height/ 2) / np.tan(np.deg2rad(fov))
        c_x = im_width / 2
        c_y = im_height / 2
        self.K = np.matrix([[f_pixels, 0, c_x], [0, f_pixels, c_y], [0, 0, 1]]).astype(float)
        return self.K

    def set_gesture_origin(self, origin_3d, H_gesture2world):
        """
        Sets class variables related to the gesture origin and pointing direction
        :param origin_3d: [4x1 np matrix] - homogenous vector of the position of the origin of the gesture in 3d real space camera frame
        :param H_gesture2world: [4 x4 np matrix] - Homogenous matrix to get from gesture frame to world frame
        :return:
        """
        # Set origin position in 3d space and convert to pixel space for drawing
        self.gesture_origin_3d = origin_3d
        self.gesture_origin_pixels = self.homgenous_3d_to_camera_pixels(origin_3d)

        # Set gesture to world R and H matricies
        self.gesture_R = H_gesture2world[0:3,0:3]
        self.gesture_H = H_gesture2world

        # Project a point 20cm in the direction of the point in camera space
        pointing_vector_bf = np.matrix([33, 0, 0]).T
        point_line = self.gesture_H @ self.make_homogenous(pointing_vector_bf)
        self.point_line = self.homgenous_3d_to_camera_pixels(point_line)

    def make_homogenous(self, vector):
        """
        turns a 3x1 vector into a homogenous vector
        :param vector: [3x1] np matrix
        :return: vector_out [4x1] np matrix
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
        """
        Converts 3d homogenous vector to 2d pixel coordinates
        :param vector_h3d: [4x1] np matrix - pre_rotated homogenous vector
        :return:vector_2d: [2x1 np matrix] - 2d vector of vector in pixel space
        """
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

    #default Found to False
    params.arm.is_found = False
    params.hand.is_found = False
    for game in params.game_list:
        game.is_found = False
    #  Use standard Aruco library
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    # Detect aruco markers in image
    markers_pos, markers_id, rejected = cv2.aruco.detectMarkers(image_frame, aruco_dict)
    # Set Poses for each marker found
    if markers_id is not None:

        for marker_idx, marker_id in enumerate(markers_id):
            marker_id = marker_id[0]
            marker_pos = markers_pos[marker_idx]

            # Set arm pose if marker is found
            if marker_id == params.arm.aruco_id:
                params.arm.set_pose(marker_pos, params)
                params.arm.is_found = True

            # Set hand pose if marker is found
            elif marker_id == params.hand.aruco_id:
                params.hand.set_pose(marker_pos, params)
                params.hand.is_found = True

            # Loop through scene objects and set pose for each object detected
            else:
                for game in params.game_list:
                    if marker_id == game.aruco_id:
                        game.set_pose(marker_pos, params)
                        game.is_found = True




def calculate_arm_pose(params):
    """
    Calculates Pose of gesture (origin and pointing vector) and stores them in param
    Note: aruco_pose must first be called to find pose of aruco tags
    :param params:Parameter class object with information about the image and scene
    :return: unit vector and 3d origin point of arm

    """

    # Check if
    # generate unit vector
    unit_vector_x_bf = np.matrix([-1, 0, 0]).T
    origin_body_frame = np.matrix([0, 0, 0, 1]).T

    # if params.arm.is_found and params.hand.is_found is not None:
    #     # TODO handle case where hand is present
    #     print('case where hand is present is not defined yet')
    #
    # elif params.hand.is_found is not None:
    #     # TODO handle case where hand is present
    #     print('case where hand is present is not defined yet')

    if params.arm.is_found is not None:
        # Find arm unit vector
        # arm_vector_x = params.arm.R @ unit_vector_x_bf
        # arm_vector_x = arm_vector_x / np.linalg.norm(arm_vector_x)

        # Find arm position in real 3d space
        arm_origin = params.arm.H_object2world @ origin_body_frame
        params.set_gesture_origin(arm_origin, params.arm.H_object2world)
    else:
        print('No gesture is detected')
        # assert(params.arm.R_object2world is not None or params.hand.R_object2world is not None)


def find_salient_object(params):
    """
    Finds which game has the most deictic saliency
    Note: calculate_arm_pose must first be called to find gesture position and orientation in params
    :param params: Scene parameter object which contains scene meta info
    :return: index of object that is being pointed to
    """


    r_norm_min = 10000
    game_min = None

    if params.arm.is_found is False:
        params.salient_game = None
        return game_min, r_norm_min

    #Find rotation matrix beween ideal gesture to object and actual gesture
    for game in params.game_list:
        if game.is_found is False:
            # print(game.name, ' is not found')
            continue

        # Find the rotation matrix between the gesture and the game unit vectors
        game.calculate_gesture_to_object_rotation(params.gesture_origin_3d,
                                                  params.gesture_R * np.matrix([1, 0, 0]).T)

        # Convert rotation matricies to rodregeues vectors (k*theta) for comparison
        r_vec, _ = cv2.Rodrigues(game.R_gesture2Object)
        r_norm = np.linalg.norm(r_vec)
        if r_norm < r_norm_min:
            game_min = game
            r_norm_min = r_norm
    if r_norm_min < 0.3:
        params.salient_game = game_min
    return game_min, r_norm_min


def draw_first_last_connected(img, points):
    for idx in range(len(points)):
        cv2.line(img, points[idx-1].astype(int), points[idx].astype(int), (0, 255, 0), thickness=20)