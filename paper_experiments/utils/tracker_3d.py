# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import pdb
from . import double_measurement_kf
from . import linear_assignment
from . import iou_matching
from .track_3d import Track_3d
from . import JPDA_matching
from . import tracking_utils
import math
from nn_matching import NearestNeighborDistanceMetric

class Tracker_3d:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : EKF.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, max_age=30, n_init=3,
                 JPDA=False, m_best_sol=1, assn_thresh=0.0,
                 matching_strategy=None, appearance_model = None,
                 gate_full_state=False, lstm = None, cuda = False, calib=None, omni=False,
                 kf_vel_params=(1./20, 1./160, 1, 1, 2), dummy_node_cost=0.2, nn_budget = None, use_imm=False,
                 markov=(0.9, 0.7), uncertainty_limit=1.8, optical_flow=False, gate_limit=400):

        self.metric = NearestNeighborDistanceMetric("euclidean", nn_budget)
        self.max_age = max_age
        self.n_init = n_init
        self.kf = double_measurement_kf.KF_3D(calib, *kf_vel_params, omni=omni)
        self.tracks = []
        self._next_id = 1
        self.JPDA = JPDA
        self.m_best_sol = m_best_sol
        self.assn_thresh = assn_thresh
        self.matching_strategy = matching_strategy
        self.gate_only_position = not gate_full_state
        self.lstm = lstm
        self.cuda = cuda
        self.dummy_node_cost = dummy_node_cost
        self.appearance_model = appearance_model

    # @profile
    def gated_metric(self, tracks, dets, track_indices, detection_indices, compare_2d=None):
        targets = np.array([tracks[i].track_id for i in track_indices])
        if not compare_2d and self.metric.check_samples(targets):
            compare_2d = True
        if compare_2d:
            features = np.array([dets[i].appearance_feature for i in detection_indices])
        else:
            features = np.array([dets[i].feature for i in detection_indices])
        #cost_matrix = self.metric.distance(features, targets, compare_2d)
        cost_matrix_appearance = self.metric.distance_torch(features, targets, compare_2d)
        use_3d = True
        for i in detection_indices:
            if dets[i].box_3d is None:
                use_3d = False
                break
        if use_3d:
            cost_matrix_iou = iou_matching.iou_cost(tracks, dets, track_indices, detection_indices, use3d=use_3d)
        else:
            cost_matrix_iou = np.ones(cost_matrix_appearance.shape)
        kf = self.kf
        dets_for_gating = dets

        gate_mask = linear_assignment.gate_cost_matrix(
            kf, tracks, dets_for_gating, track_indices,
            detection_indices, only_position=self.gate_only_position, use3d=use_3d)
        cost_matrix = np.dstack((cost_matrix_appearance, cost_matrix_iou))

        return cost_matrix, gate_mask

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    # @profile
    def update(self, input_img, detections, compare_2d):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """

        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, compare_2d)

        # update filter for each assigned track
        # Only do this for non-JPDA because in JPDA the kf states are updated
        # during the matching process
        
        if not self.JPDA:
            # Map matched tracks to detections
            track_detection_map = {t:d for (t,d) in matches}
            # Map unmatched tracks to -1 for no detection
            for t in unmatched_tracks:
                track_detection_map[t] = -1
            
            for track_idx, detection_idx in matches:
                self.tracks[track_idx].update(self.kf, detections,
                        detection_idx=detection_idx, JPDA=self.JPDA, 
                        cur_frame = self.cur_frame, appearance_model = self.appearance_model, 
                        lstm = self.lstm)
        # update track state for unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        self.prune_tracks()
        # create new tracks
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

         # Update distance metric.
        active_targets = [t.track_id for t in self.tracks]
        features, features_2d, targets, targets_2d = [], [], [], []
        for track in self.tracks:
            features += track.features
            features_2d += track.features_2d
            targets += [track.track_id for _ in track.features]
            targets_2d += [track.track_id for _ in track.features_2d]
            track.features = []
            track.features_2d = []

        self.metric.partial_fit(
            np.asarray(features), np.asarray(features_2d), np.asarray(targets), np.asarray(targets_2d), active_targets)

    # @profile
    def _match(self, detections, compare_2d):

        # Associate confirmed tracks using appearance features.
        if self.JPDA:
            # Only run JPDA on confirmed tracks
            marginalizations = \
            linear_assignment.JPDA(self.gated_metric, self.dummy_node_cost, self.tracks, \
                detections, compare_2d=compare_2d)

            jpda_matcher = JPDA_matching.Matcher(
                detections, marginalizations, range(len(self.tracks)),
                self.matching_strategy, assignment_threshold=self.assn_thresh)
            matches_a, unmatched_tracks_a, unmatched_detections = jpda_matcher.match()

            # Map matched tracks to detections
            track_detection_map = {t:d for (t,d) in matches_a}

            # Map unmatched tracks to -1 for no detection
            for t in unmatched_tracks_a:
                track_detection_map[t] = -1

            # udpate Kalman state
            if marginalizations.shape[0] > 0:
                for i in range(len(self.tracks)):
                    self.tracks[i].update(self.kf, detections,
                        marginalization=marginalizations[i,:], detection_idx=track_detection_map[i], 
                        JPDA=self.JPDA, lstm = self.lstm)

        else:
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    self.gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks, compare_2d = compare_2d, detections_3d=detections_3d)

        return matches_a, unmatched_tracks_a, unmatched_detections

    def _initiate_track(self, detection):
        if detection.box_3d is None:
            return
        mean, covariance = self.kf.initiate(detection.box_3d)
        self.tracks.append(Track_3d(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            feature=detection.feature, appearance_feature = detection.appearance_feature,
            cuda = self.cuda, lstm = self.lstm))
        self._next_id += 1
    
    def prune_tracks(self):

        # for track in self.tracks:
        #     # Check if track is leaving
        #     predicted_mean = self.kf.predict_mean(track.mean)
        #     predicted_cov = track.covariance
        #     predicted_pos = predicted_mean[:2]
        #     predicted_vel = predicted_mean[4:6]
        #     predicted_pos[0] -= w/2
        #     predicted_pos[1] -= h/2

        #     cos_theta = np.dot(predicted_pos, predicted_vel)/(np.linalg.norm(predicted_pos)*
        #                                             np.linalg.norm(predicted_vel) + 1e-6)
        #     predicted_pos[0] += w/2
        #     predicted_pos[1] += h/2
        #     # Thresholds for deciding whether track is outside image
        #     BORDER_VALUE = 0
        #     if (cos_theta > 0 and
        #         (predicted_pos[0] - track.mean[2]/2<= BORDER_VALUE or
        #         predicted_pos[0] + track.mean[2]/2 >= w - BORDER_VALUE)):
        #         if track.is_exiting() and not track.matched:
        #             track.delete_track()
        #         else:
        #             track.mark_exiting()
            # Check if track is too uncertain
            # cov_axis,_ = np.linalg.eigh(predicted_cov)
            # if np.abs(np.sqrt(cov_axis[-1]))*6 > self.uncertainty_limit*np.linalg.norm(predicted_mean[2:4]):
            #    track.delete_track()
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
