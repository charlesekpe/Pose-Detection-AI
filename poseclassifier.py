import numpy as np
class PoseClassifier(object):
    """Classifies pose landmarks."""
    def __init__(self, 
        landmarks=None,
        n_landmarks=33,
        threshold=0.16,
        point='pinky_1'):

        '''
        Initializes pose classifier variables
        @Parameters:
            landmarks: 2-d array, list or tuple
            A list, array or tuple of floats representing the location of keypoints of 33 body parts of the image
            point: str
            One of [pinky_1, thumb_2, index_1], representing the finger of the hand we use as end of the hand
            n_landmarks: int
            number of arrays in landmarks
            threshold: float
            float in range [0,1] to calculate the minimum distance between two landmark point

        '''
        self.landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
            ]
        self.landmarks = landmarks
        self._n_landmarks = n_landmarks
        self.threshold = threshold
        self.point = point
        self.landmark_dict = {i:j for j, i in enumerate(self.landmark_names)}
    
    def _distance(self, a, b):
        eucl_dist = (((a[0]-b[0])**2) + ((a[1] - b[1])**2))**0.5
        return eucl_dist
    
    def right_ear(self, landmarks):
        if landmarks is None:
            return False
        lh, re = self.landmark_dict['left_'+ self.point],  self.landmark_dict['right_ear']
        dist = self._distance(landmarks[lh], landmarks[re])
        return dist
    
    def left_ear(self, landmarks):
        if landmarks is None:
            return False
        rh, le = self.landmark_dict['right_'+ self.point],  self.landmark_dict['left_ear']
        dist = self._distance(landmarks[rh], landmarks[le])
        return dist

    def right_shoulder(self, landmarks):
        if landmarks is None:
            return False
        lh, rs = self.landmark_dict['left_'+ self.point],  self.landmark_dict['right_shoulder']
        dist = self._distance(landmarks[lh], landmarks[rs])
        return dist

    def left_shoulder(self, landmarks):
        if landmarks is None:
            return False
        rh, ls = self.landmark_dict['right_'+ self.point],  self.landmark_dict['left_shoulder']
        dist = self._distance(landmarks[rh], landmarks[ls])
        return dist

    def right_hip(self, landmarks):
        if landmarks is None:
            return False
        lh, rp = self.landmark_dict['left_'+ self.point],  self.landmark_dict['right_hip']
        dist = self._distance(landmarks[lh], landmarks[rp])
        return dist

    def left_hip(self, landmarks):
        if landmarks is None:
            return False
        rh, lp = self.landmark_dict['right_'+ self.point],  self.landmark_dict['left_hip']
        dist = self._distance(landmarks[rh], landmarks[lp])
        return dist

    def right_knee(self, landmarks):
        if landmarks is None:
            return False
        lh, rk = self.landmark_dict['left_'+ self.point],  self.landmark_dict['right_knee']
        dist = self._distance(landmarks[lh], landmarks[rk])
        return dist

    def left_knee(self, landmarks):
        if landmarks is None:
            return False
        rh, lk = self.landmark_dict['right_'+ self.point],  self.landmark_dict['left_knee']
        dist = self._distance(landmarks[rh], landmarks[lk])
        return dist


    def determine_pose(self, landmarks):
        self.landmarks = landmarks
        assert len(self.landmarks) == self._n_landmarks, 'Unexpected landmarks shape: {}'.format(len(landmarks))
        pose_calc = [
            self.right_ear,
            self.left_ear,
            self.right_shoulder,
            self.left_shoulder,
            self.right_hip,
            self.left_hip,
            self.right_knee,
            self.left_knee
        ]
        pose_dist = [float(f(landmarks)) for f in pose_calc]
        '''
        If minimum value in the list is less than threshold, 
        return a sparse binary array with one at the minimum position
        '''
        if np.min(pose_dist)<=self.threshold:
            pose_array = np.zeros(8)
            pose_array[np.argmin(pose_dist)]=1
        else:
            pose_array = np.zeros(8)
        return pose_array
