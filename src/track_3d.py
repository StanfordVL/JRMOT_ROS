# vim: expandtab:ts=4:sw=4
import numpy as np
import pdb
import torch

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track_3d:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, appearance_feature = None, cuda = False, lstm = None):

        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.cuda = cuda
        self.state = TrackState.Tentative
        self.features = []
        self.features_2d = []
        self.hidden = None
        if lstm is None:
            self.features.append(feature)
            self.features_2d.append(appearance_feature)
        else:
            self.feature_update(feature, appearance_feature, lstm)
        self.first_detection = mean[:7]
        self._n_init = n_init
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        self._max_age = max_age
        self.matched = True
        self.exiting = False
        self.last_box = None


    def to_tlwh3d(self):
        """Get current position in bounding box format `(box center of bottom face [x, y, z], l, w, h)`.

        Returns
        -------
        ndarray
            The bounding box.

        """

        if self.last_box is not None:
            return self.last_box.box_3d
        else:
            return self.mean[[0,1,2,3,4,5,6]].copy()

    def to_tlwh(self, kf):
        """Get current position in bounding box format `(box center of bottom face [x, y, z], l, w, h)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        corner_points, _ = kf.calculate_corners(self.mean)
        min_x, min_y = np.amin(corner_points, axis = 0)[:2]
        max_x, max_y = np.amax(corner_points, axis = 0)[:2]
        ret = np.array([min_x, min_y, max_x - min_x, max_y - min_y])
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    # @profile
    def update(self, kf, detection, compare_2d=False,
                marginalization=None, detection_idx=None, JPDA=False, lstm = None):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        if JPDA:

            detections_2d = [det.tlwh for det in detection]
            if compare_2d:
                detections_3d = None
            else:
                detections_3d = [np.copy(det.box_3d) for det in detection]
                for det in detections_3d:
                    if det[6] - self.mean[6] > np.pi:
                        det[6] -= 2 * np.pi
                    elif det[6] - self.mean[6] < -np.pi:
                        det[6] += 2*np.pi
            self.mean, self.covariance, self.mean_post_3d = kf.update(
                self.mean, self.covariance, detections_2d, detections_3d, marginalization, JPDA)
            self.mean[6] = self.mean[6] % (2 * np.pi)
            self.feature_update(detection, detection_idx, lstm)
            if np.argmax(marginalization) != 0:
                self.matched=True
            else:
                self.matched=False
            if detection_idx < 0:
                self.last_box = None
                return
            self.hits += 1
            self.time_since_update = 0
            detection = detection[detection_idx]
            self.last_box = detection
        else:
            detection = detection[detection_idx]
            detections_3d = detections_3d[detection_idx]
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, detection.tlwh, detections_3d)

        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def feature_update(self, detections, detection_idx, lstm, JPDA=False, marginalization=None):
        if JPDA:
            features=[d.feature for d in detections]
            appearance_features=[d.appearance_feature for d in detections]
            if len([i for i in features if i is None])==0:
                combined_feature=np.sum(np.array(features).reshape(len(features), -1)
                                        *marginalization[1:].reshape(-1, 1), axis=0).astype(np.float32)
                self.features.append(combined_feature)
            if len([i for i in appearance_features if i is None])==0:
                combined_feature=np.sum(
                                np.array(appearance_features).reshape(len(appearance_features), -1)
                                *marginalization[1:].reshape(-1, 1), axis=0).astype(np.float32)
                self.features_2d.append(combined_feature)
        else:
            feature = detections[detection_idx].feature
            appearance_feature = detections[detection_idx].appearance_feature
            if feature is not None:
                if lstm is not None:
                    input_feature = torch.Tensor(feature).type(self.tensor)
                    input_feature = input_feature.unsqueeze(0)
                    with torch.no_grad():
                        if self.hidden is None:
                            output_feature, self.hidden = lstm(input_feature)
                        else:
                            output_feature, self.hidden = lstm(input_feature, self.hidden)
                    output_feature = output_feature.cpu().numpy().squeeze(0)
                else:
                    output_feature = feature
                self.features.append(output_feature)
            if appearance_feature is not None:
                self.features_2d.append(appearance_feature)

    def get_cov(self):
        xyz_cov = self.covariance[:3, :3]
        theta_cov_1 = self.covariance[7, :3]
        theta_cov_2 = self.covariance[7, 7]
        out_cov = np.zeros((6, 6))
        out_cov[:3,:3] = xyz_cov
        out_cov[5, :3] = theta_cov_1
        out_cov[:3, 5] = theta_cov_1
        out_cov[5, 5] = theta_cov_2
        return out_cov