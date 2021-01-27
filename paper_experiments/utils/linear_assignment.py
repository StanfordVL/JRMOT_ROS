# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import EKF
import pdb
from mbest_ilp import new_m_best_sol
from multiprocessing import Pool
from functools import partial
#from mbest_ilp import m_best_sol as new_m_best_sol

INFTY_COST = 1e+5

def min_marg_matching(marginalizations, track_indices=None, max_distance=1):
    cost_matrix = 1 - marginalizations
    num_tracks, num_detections = cost_matrix.shape
    if track_indices is None:
        track_indices = np.arange(num_tracks)

    detection_indices = np.arange(num_detections-1)

    if num_tracks == 0 or num_detections == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    extra_dummy_cols = np.tile(cost_matrix[:,0,np.newaxis], (1, num_tracks-1))
    expanded_cost_matrix = np.hstack((extra_dummy_cols, cost_matrix))
    indices = linear_assignment(expanded_cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []

    # gather unmatched detections (new track)
    for col, detection_idx in enumerate(detection_indices):
        if col+num_tracks not in indices[:, 1]:
            unmatched_detections.append(detection_idx)

    # gather unmatched tracks (no detection)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)

    # thresholding and matches
    for row, col in indices:

        track_idx = track_indices[row]
        detection_idx = col - num_tracks
        if detection_idx < 0:
            unmatched_tracks.append(track_idx)
            continue

        if expanded_cost_matrix[row, col] > max_distance:
            # apply thresholding
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            # associate matches
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections

def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None, compare_2d = False, detections_3d=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices, compare_2d, detections_3d)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    #print("\n\nCascade Cost Matrix: ", cost_matrix)

    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []

    # gather unmatched detections (new track)
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)

    # gather unmatched trackes (no detection)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)

    # thresholding and matches
    for row, col in indices:

        track_idx = track_indices[row]
        detection_idx = detection_indices[col]

        if cost_matrix[row, col] > max_distance:
            # apply thresholding
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            # associate matches
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections

# @profile
def JPDA(
        distance_metric, dummy_node_cost_app, dummy_node_cost_iou, tracks, detections, track_indices=None,
        detection_indices=None, m=1, compare_2d = False, windowing = False):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return np.zeros((0, len(detections) + 1))  # Nothing to match.
    cost_matrix, gate_mask = distance_metric(
        tracks, detections, track_indices, detection_indices, compare_2d)
    
    num_tracks, num_detections = cost_matrix.shape[0], cost_matrix.shape[1]
    cost_matrix[gate_mask] = INFTY_COST
    
    # print("\nIOU Cost Matrix:", cost_matrix[:,:,0])
    # print("App:", cost_matrix[:,:,1])

    clusters = find_clusters(cost_matrix[:,:,0], INFTY_COST - 0.0001)
    # print('\n', clusters)

    jpda_output = []
    for cluster in clusters:
        jpda_output.append(get_JPDA_output(cluster, cost_matrix, dummy_node_cost_app, dummy_node_cost_iou, INFTY_COST - 0.0001, m))
    if not jpda_output:
        mc = np.zeros((num_tracks, num_detections + 1))
        mc[:, 0] = 1
        return mc
    assignments, assignment_cost = zip(*jpda_output)
    assignments = np.vstack([item for sublist in assignments for item in sublist])
    assignment_cost = np.array([item for sublist in assignment_cost for item in sublist])

    marginalised_cost = np.sum(assignments*np.exp(-np.expand_dims(assignment_cost, 1)), axis = 0)
    marginalised_cost = np.reshape(marginalised_cost, (num_tracks, num_detections+1))
    # print('\n', marginalised_cost)
    return marginalised_cost

def calculate_entropy(matrix, idx, idy):
    mask = np.ones(matrix.shape)
    mask[idx, idy] = 0
    entropy = matrix/np.sum(mask*matrix, axis=1, keepdims=True)
    entropy = (-entropy*np.log(entropy)) * mask
    entropy = np.mean(np.sum(entropy, axis=1))
    return entropy

def get_JPDA_output(cluster, cost_matrix, dummy_node_cost_app, dummy_node_cost_iou, cutoff, m):
    if len(cluster[1]) == 0:
        assignment = np.zeros((cost_matrix.shape[0], cost_matrix.shape[1]+1))
        assignment[cluster[0], 0] = 1
        assignment = assignment.reshape(1,-1)
        return [assignment], np.array([0])
    
    new_cost_matrix_appearance = np.reshape(cost_matrix[np.repeat(cluster[0], len(cluster[1])), 
                                        np.tile(cluster[1] - 1, len(cluster[0])), 
                                        [0]*(len(cluster[1])*len(cluster[0]))], 
                                        (len(cluster[0]), len(cluster[1])))
    new_cost_matrix_iou = np.reshape(cost_matrix[np.repeat(cluster[0], len(cluster[1])), np.tile(cluster[1] - 1, len(cluster[0])), 1], 
                (len(cluster[0]), len(cluster[1])))
    idx_x, idx_y = np.where(new_cost_matrix_appearance > cutoff)
    appearance_entropy = calculate_entropy(new_cost_matrix_appearance, idx_x, idx_y)
    iou_entropy = calculate_entropy(new_cost_matrix_iou, idx_x, idx_y)
    if appearance_entropy < iou_entropy:
        new_cost_matrix = new_cost_matrix_appearance
        new_cost_matrix = 2*np.ones(new_cost_matrix.shape)/(new_cost_matrix+1) - 1
        dummy_node_cost = -np.log(2/(dummy_node_cost_app+1) - 1)
    else:
        new_cost_matrix = new_cost_matrix_iou
        new_cost_matrix[new_cost_matrix==1] -= 1e-3
        new_cost_matrix = 1 - new_cost_matrix
        dummy_node_cost = -np.log(1-dummy_node_cost_iou)
    new_cost_matrix = -np.log(new_cost_matrix)
    new_cost_matrix[idx_x, idx_y] = cutoff
    if len(cluster[0]) == 1:
        new_cost_matrix = np.concatenate([np.ones((new_cost_matrix.shape[0], 1))*dummy_node_cost, new_cost_matrix], axis = 1)
        total_cost = np.sum(np.exp(-new_cost_matrix))
        new_assignment = np.zeros((cost_matrix.shape[0], cost_matrix.shape[1]+1))
        new_assignment[np.repeat(cluster[0], len(cluster[1])+1), np.tile(
                        np.concatenate([np.zeros(1, dtype = np.int32), cluster[1]]), len(cluster[0]))] = np.exp(-new_cost_matrix)/total_cost
        new_assignment = new_assignment.reshape(1, -1)
        return  [new_assignment], np.array([0])
    if new_cost_matrix.ndim <= 1:
        new_cost_matrix = np.expand_dims(new_cost_matrix, 1)

    # print(new_cost_matrix)
    assignments, assignment_cost = new_m_best_sol(new_cost_matrix, m, dummy_node_cost)
    offset = np.amin(assignment_cost)
    assignment_cost -= offset
    new_assignments = []
    total_cost = np.sum(np.exp(-assignment_cost))
    for assignment in assignments:
        new_assignment = np.zeros((cost_matrix.shape[0], cost_matrix.shape[1]+1))
        new_assignment[np.repeat(cluster[0], len(cluster[1])+1), np.tile(
                    np.concatenate([np.zeros(1, dtype = np.int32), cluster[1]]), len(cluster[0]))] = \
                                                assignment/total_cost
        new_assignments.append(new_assignment.reshape(1, -1))
    return new_assignments, assignment_cost


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None, compare_2d = False, detections_3d=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections, compare_2d, detections_3d=detections_3d)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections

# @profile
def gate_cost_matrix(
        kf, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False, use3d=False, windowing = False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
   
    # assert (len(track_indices) == cost_matrix.shape[0]), "Cost matrix shape does not match track indices"
    # assert (len(detection_indices) == cost_matrix.shape[1]), "Cost matrix shape does match detection indices"

    if len(track_indices) == 0 or len(detection_indices) == 0:
        return None

    if use3d:
        measurements = np.array([det.box_3d for i, det in enumerate(detections) if i in detection_indices])
    else:
        measurements = np.asarray(
            [detections[i].to_xywh() for i in detection_indices])
    if use3d and only_position:
        gating_dim = 3
    elif use3d:
        gating_dim =  measurements.shape[1]
    elif only_position:
        gating_dim = 2
    else:
        gating_dim =  measurements.shape[1]
    gating_threshold = EKF.chi2inv95[gating_dim]
    gate_mask = []
    for track_idx in track_indices:
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, use3d)
        gated_set = gating_distance > gating_threshold
        if np.all(gated_set):
            gated_set = gating_distance > gating_threshold * 3
        # print(track.track_id, gating_threshold, gating_distance)
        gate_mask.append(gated_set)
        # print(gated_set)
    return np.vstack(gate_mask)

def find_clusters(cost_matrix, cutoff):
    num_tracks, _ = cost_matrix.shape
    clusters = []
    total_tracks = 0
    total_detections = 0
    all_tracks = set(range(num_tracks))
    all_visited_tracks = set()
    while total_tracks < num_tracks:
        visited_detections = set()
        visited_tracks = set()
        potential_track = next(iter(all_tracks - all_visited_tracks))
        potential_tracks = set()
        potential_tracks.add(potential_track)
        while potential_tracks:
            current_track = potential_tracks.pop()
            visited_detections.update((np.where(cost_matrix[current_track] < cutoff)[0])+1)
            visited_tracks.add(current_track)
            for detection in visited_detections:
                connected_tracks = np.where(cost_matrix[:, detection - 1] < cutoff)[0]
                for track in connected_tracks:
                    if track in visited_tracks or track in potential_tracks:
                        continue
                    potential_tracks.add(track)
        total_tracks += len(visited_tracks)
        total_detections += len(visited_detections)
        all_visited_tracks.update(visited_tracks)
        clusters.append((np.array(list(visited_tracks), dtype = np.int32), np.array(list(visited_detections), dtype = np.int32)))
    return clusters
