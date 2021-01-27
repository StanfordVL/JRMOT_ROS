# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import pdb
from . import kf_2d, kf_3d, double_measurement_kf, imm
from . import linear_assignment
from . import iou_matching
from .track import Track
from . import JPDA_matching
from . import tracking_utils
import math
from nn_matching import NearestNeighborDistanceMetric
import cv2


class Tracker:
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

    def __init__(self, max_age=5, n_init=3,
                 JPDA=False, m_best_sol=1, assn_thresh=0.0,
                 matching_strategy=None,
                 kf_appearance_feature=None,
                 gate_full_state=False, lstm = None, cuda = False, appearance_model = None,
                 calib = None, kf_vel_params=(1./20, 1./160, 1, 1, 2), dummy_node_cost_iou=0.4, dummy_node_cost_app=0.2, nn_budget = None, use_imm=False, kf_walk_params=(1./20, 1./160, 1, 1, 2),
                 markov=(0.9, 0.7), uncertainty_limit=1.8, optical_flow=False, gate_limit=400):

        self.max_age = max_age
        self.n_init = n_init
        self.metric = NearestNeighborDistanceMetric("euclidean", nn_budget)
        if not use_imm:
            self.kf = kf_2d.KalmanFilter2D(*kf_vel_params, gate_limit)
            self.use_imm = False
        else:
            self.kf = imm.IMMFilter2D(kf_vel_params, kf_walk_params, markov=markov)
            self.use_imm = True
        self.tracks = []
        self._next_id = 1
        self.JPDA = JPDA
        self.m_best_sol = m_best_sol
        self.assn_thresh = assn_thresh
        self.matching_strategy = matching_strategy
        self.kf_appearance_feature = kf_appearance_feature
        self.gate_only_position = not gate_full_state
        self.lstm = lstm
        self.cuda = cuda
        self.dummy_node_cost_app = dummy_node_cost_app
        self.dummy_node_cost_iou = dummy_node_cost_iou
        self.appearance_model = appearance_model
        self.prev_frame = None
        self.uncertainty_limit = uncertainty_limit
        self.optical_flow = optical_flow

    # @profile
    def gated_metric(self, tracks, dets, track_indices, detection_indices, compare_2d = False):
        targets = np.array([tracks[i].track_id for i in track_indices])
        if not compare_2d and self.metric.check_samples(targets):
            compare_2d = True
        if compare_2d:
            features = np.array([dets[i].appearance_feature for i in detection_indices])
        else:
            features = np.array([dets[i].feature for i in detection_indices])
        #cost_matrix = self.metric.distance(features, targets, compare_2d)
        cost_matrix_appearance = self.metric.distance_torch(features, targets, compare_2d)
        cost_matrix_iou = iou_matching.iou_cost(tracks, dets, track_indices, detection_indices)

        gate_mask = linear_assignment.gate_cost_matrix(
            self.kf, tracks, dets, track_indices,
            detection_indices, only_position=self.gate_only_position)
        cost_matrix = np.dstack((cost_matrix_appearance, cost_matrix_iou))

        return cost_matrix, gate_mask

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    # @profile
    def update(self, cur_frame, detections, compare_2d = False):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        
        self.cur_frame = cv2.cvtColor((255*cur_frame).permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2GRAY)

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

        # create new tracks
        self.prune_tracks()
        flow = None
        if unmatched_detections:
            if self.optical_flow and self.prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(self.prev_frame, self.cur_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], flow)

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
        self.prev_frame = self.cur_frame

    # @profile
    def _match(self, detections, compare_2d):

        # Associate all tracks using combined cost matrices.
        if self.JPDA:
            # Run JPDA on all tracks
            marginalizations = \
            linear_assignment.JPDA(self.gated_metric, self.dummy_node_cost_app, self.dummy_node_cost_iou, self.tracks, \
                detections, m=self.m_best_sol, compare_2d = compare_2d)
            # for track in self.tracks: #TODO: REMOVE
            #     print(track.track_id)
            # print(marginalizations)

            jpda_matcher = JPDA_matching.Matcher(
                detections, marginalizations, range(len(self.tracks)),
                self.matching_strategy, assignment_threshold=self.assn_thresh)
            matches_a, unmatched_tracks_a, unmatched_detections = jpda_matcher.match()

            # Map matched tracks to detections
            # Map matched tracks to detections
            track_detection_map = {t:d for (t,d) in matches_a}
            # Map unmatched tracks to -1 for no detection
            for t in unmatched_tracks_a:
                track_detection_map[t] = -1
            # update Kalman state
            if marginalizations.shape[0] > 0:
                for i in range(len(self.tracks)):
                    self.tracks[i].update(self.kf, detections,
                        marginalization=marginalizations[i,:], detection_idx=track_detection_map[i], 
                        JPDA=self.JPDA, cur_frame = self.cur_frame, appearance_model = self.appearance_model, lstm = self.lstm)
        else:
            confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    self.gated_metric, self.dummy_node_cost_iou, self.max_age,
                    self.tracks, detections, confirmed_tracks, compare_2d = compare_2d)
        return matches_a, unmatched_tracks_a, unmatched_detections

    def _initiate_track(self, detection, flow=None):
        if self.use_imm:
            mean, covariance, model_probabilities = self.kf.initiate(detection.to_xywh(), flow)
        else:
            mean, covariance = self.kf.initiate(detection.to_xywh(), flow)
            model_probabilities = None

        self.tracks.append(Track(
            mean, covariance, model_probabilities, self._next_id, self.n_init, self.max_age,
            kf_appearance_feature = self.kf_appearance_feature,
            feature=detection.feature, appearance_feature = detection.appearance_feature,
            cuda = self.cuda, lstm = self.lstm, last_det = detection))
        self._next_id += 1
    
    def prune_tracks(self):
        h, w = self.cur_frame.shape
        for track in self.tracks:
            # Check if track is leaving
            if self.use_imm:
                predicted_mean, predicted_cov = self.kf.combine_states(track.mean, track.covariance, track.model_probabilities) #TODO: This doesn't predict. Mean should def predict
            else:
                predicted_mean = self.kf.predict_mean(track.mean)
                predicted_cov = track.covariance
            predicted_pos = predicted_mean[:2]
            predicted_vel = predicted_mean[4:6]
            predicted_pos[0] -= w/2
            predicted_pos[1] -= h/2

            cos_theta = np.dot(predicted_pos, predicted_vel)/(np.linalg.norm(predicted_pos)*
                                                    np.linalg.norm(predicted_vel) + 1e-6)
            predicted_pos[0] += w/2
            predicted_pos[1] += h/2
            # Thresholds for deciding whether track is outside image
            BORDER_VALUE = 0
            if (cos_theta > 0 and
                (predicted_pos[0] - track.mean[2]/2<= BORDER_VALUE or
                predicted_pos[0] + track.mean[2]/2 >= w - BORDER_VALUE)):
                if track.is_exiting() and not track.matched:
                    track.delete_track()
                else:
                    track.mark_exiting()
            # Check if track is too uncertain
            # cov_axis,_ = np.linalg.eigh(predicted_cov)
            # if np.abs(np.sqrt(cov_axis[-1]))*6 > self.uncertainty_limit*np.linalg.norm(predicted_mean[2:4]):
            #    track.delete_track()
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
