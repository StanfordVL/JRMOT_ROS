# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from linear_assignment import min_marg_matching
import pdb


def get_unmatched(all_idx, matches, i, marginalization=None):
    assigned = [match[i] for match in matches]
    unmatched = set(all_idx) - set(assigned)
    if marginalization is not None:
        # from 1 for dummy node
        in_gate_dets = np.nonzero(np.sum(
            marginalization[:, 1:], axis=0))[0].tolist()
        unmatched = [d for d in unmatched if d not in in_gate_dets]
    return list(unmatched)


class Matcher:

    def __init__(self, detections, marginalizations, confirmed_tracks,
                 matching_strategy,
                 assignment_threshold=None):
        self.detections = detections
        self.marginalizations = marginalizations
        self.confirmed_tracks = confirmed_tracks
        self.assignment_threshold = assignment_threshold
        self.detection_indices = np.arange(len(detections))
        self.matching_strategy = matching_strategy

    def match(self):
        self.get_matches()
        self.get_unmatched_tracks()
        self.get_unmatched_detections()
        return self.matches, self.unmatched_tracks, self.unmatched_detections

    def get_matches(self):

        if self.matching_strategy == "max_and_threshold":
            self.max_and_threshold_matching()
        elif self.matching_strategy == "hungarian":
            self.hungarian()
        elif self.matching_strategy == "max_match":
            self.max_match()
        elif self.matching_strategy == "none":
            self.matches = []
        else: 
            raise Exception('Unrecognized matching strategy: {}'.
                            format(self.matching_strategy))

    def get_unmatched_tracks(self):
        self.unmatched_tracks = get_unmatched(self.confirmed_tracks,
                                              self.matches, 0)

    def get_unmatched_detections(self):
        self.unmatched_detections = get_unmatched(self.detection_indices, self.matches, 1, self.marginalizations)

    def max_match(self):
        self.matches = []
        if self.marginalizations.shape[0] == 0:
            return

        detection_map = {}
        for i, track_idx in enumerate(self.confirmed_tracks):
            marginalization = self.marginalizations[i,:]
            detection_id = np.argmax(marginalization) - 1  # subtract one for dummy

            if detection_id < 0:
                continue

            if detection_id not in detection_map.keys():
                detection_map[detection_id] = track_idx
            else:
                cur_track = detection_map[detection_id]
                track_update = track_idx if self.marginalizations[track_idx, detection_id] > self.marginalizations[cur_track, detection_id] else cur_track
                detection_map[detection_id] = track_update
            threshold_p = marginalization[detection_id + 1]
            if threshold_p < self.assignment_threshold:
                continue

        for detection in detection_map.keys():
            self.matches.append((detection_map[detection], detection))

    def max_and_threshold_matching(self):

        self.matches = []
        if self.marginalizations.shape[0] == 0:
            return

        for i, track_idx in enumerate(self.confirmed_tracks):
            marginalization = self.marginalizations[i,:]
            detection_id = np.argmax(marginalization) - 1  # subtract one for dummy

            if detection_id < 0:
                continue

            threshold_p = marginalization[detection_id + 1]
            if threshold_p < self.assignment_threshold:
                continue

            self.matches.append((track_idx, detection_id))

    def hungarian(self):
        self.matches, _, _ = min_marg_matching(self.marginalizations,
                                               self.confirmed_tracks,
                                               self.assignment_threshold)
                               