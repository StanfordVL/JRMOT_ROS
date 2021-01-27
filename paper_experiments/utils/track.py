# vim: expandtab:ts=4:sw=4
import numpy as np
import pdb
import torch
import copy

from .imm import IMMFilter2D

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


class Track:
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
    def __init__(self, mean, covariance, model_probabilities, track_id, n_init, max_age,
                 feature=None, appearance_feature = None, cuda = False, lstm = None, kf_appearance_feature=False, last_det = None):

        self.mean = mean
        self.covariance = covariance
        self.model_probabilities = model_probabilities
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
        self.kf_appearance_feature = kf_appearance_feature
        if lstm is None:
            self.features.append(feature)
            self.features_2d.append(appearance_feature)
        else:
            self.feature_update(feature, appearance_feature, lstm)
        if self.model_probabilities is not None:
            self.first_detection = mean[:,:4]
        else:
            self.first_detection = mean[:4]

        self._n_init = n_init
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        self._max_age = max_age
        self.matched = True
        self.exiting = False
        self.next_to_last_detection = None
        self.last_detection = last_det
        self.last_2d_det = last_det

    def to_tlwh(self, kf):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        if self.model_probabilities is None:
            if self.last_2d_det is not None: #TODO: This part
                # print(self.last_2d_det.to_xywh(), self.mean[:4])
                ret = self.last_2d_det.to_xywh()
            else:
                ret = self.mean[:4].copy()
        else:
            mean, _ = IMMFilter2D.combine_states(self.mean, self.covariance, self.model_probabilities)
            ret = mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh(None)
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def update_feature(self, img, appearance_model):

        x = round(self.mean[0])
        y = round(self.mean[1])
        a = self.mean[2]
        box_h = int(round(self.mean[3]))

        x1 = int(round(x - (x / 2)))
        y1 = int(round(y - (y / 2)))
        box_w = int(round(a * box_h))

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # patch = torch.Tensor(img[y1:y1+box_h, x1:x1+box_w, :]).type(Tensor).permute(2,0,1)
        patch = img[:, y1:y1+box_h, x1:x1+box_w]

        if patch is None or patch.nelement()==0:
            return None
        patch = patch.unsqueeze(0)

        with torch.no_grad():
            feature ,_ = appearance_model(patch)

            return feature.squeeze(0)

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        if self.model_probabilities is None:
            self.mean, self.covariance = kf.predict(self.mean, self.covariance, self.last_detection, self.next_to_last_detection)
        else:
            self.mean, self.covariance, self.model_probabilities = kf.predict(self.mean, self.covariance, self.model_probabilities)
        self.age += 1
        self.time_since_update += 1

    # @profile
    def update(self, kf, detection, detections_3d=None,
                marginalization=None, detection_idx=None, JPDA=False,
                cur_frame = None, appearance_model = None, lstm = None,
                only_feature=False):
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
            detections = [det.to_xywh() for det in detection]
            if self.model_probabilities is None:
                self.mean, self.covariance = kf.update(
                    self.mean, self.covariance, detections, marginalization, JPDA)
            else:
                self.mean, self.covariance, self.model_probabilities = kf.update(self.mean, self.covariance, detections, self.model_probabilities, marginalization, JPDA)
            self.feature_update(detection, detection_idx, lstm)
            if np.argmax(marginalization) != 0:
                self.matched=True
            else:
                self.matched=False
            if detection_idx < 0:
                self.last_2d_det = None
                return
            self.hits += 1
            self.time_since_update = 0
            detection = detection[detection_idx]
            self.last_2d_det = detection

        else:
            detection = detection[detection_idx]
            if self.model_probabilities is None:
                self.mean, self.covariance = kf.update(
                    self.mean, self.covariance, detection.to_xywh())
            else:
                self.mean, self.covariance, self.model_probabilities = kf.update(self.mean, self.covariance, detection.to_xyah(), self.model_probabilities)
            self.feature_update(detection.feature, detection.appearance_feature, lstm)
            self.hits += 1
            self.time_since_update = 0
        if detection.box_3d is not None:
            self.next_to_last_detection = self.last_detection
            self.last_detection = detection
        if self.age==2:
            self.update_velocity(detection.to_xywh())
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
    
    def delete_track(self):
        self.state = TrackState.Deleted
    
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def update_velocity(self, new_detection):
        if self.model_probabilities is not None:
            for kf_n in range(2):
                velocity_estimate = new_detection - self.first_detection
                self.mean[kf_n,4:] = velocity_estimate[kf_n,:4]
                # Reduce covariance of velocity by 4 times (half the standard deviation)
                self.covariance[kf_n,:,4:] /= 4
                self.covariance[kf_n,4:,:] /= 4
        else:
            velocity_estimate = new_detection - self.first_detection
            self.mean[4:] = velocity_estimate[:4]
            # Reduce covariance of velocity by 4 times (half the standard deviation)
            self.covariance[:,4:] /= 4
            self.covariance[4:,:4] /= 4

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
    
    def is_exiting(self):
        return self.exiting
    
    def mark_exiting(self):
        self.exiting = True

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
                # print("track:", self.track_id, "original", len(self.features), "2D", len(self.features_2d))
                self.features.append(output_feature)
                # diffs = [] #TODO: REMOVE
                # for i in range(len(self.features)-1):
                #     diffs.append(np.linalg.norm(self.features[i],self.features[i+1]))
                # diffs = np.asarray(diffs)
                # print("track:", self.track_id, "count:", len(self.features),"mean", np.mean(diffs), "std", np.std(diffs))
            if appearance_feature is not None:
                self.features_2d.append(appearance_feature)
