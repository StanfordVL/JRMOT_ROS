# vim: expandtab:ts=4:sw=4
import numpy as np
import pdb
import torch

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def _cosine_distance_torch(a, b, data_is_normalized=False):
    '''
    _cosine_distance but torched
    '''
    if not data_is_normalized:
        a = a / torch.norm(a, dim=1, keepdim=True)
        b = b / torch.norm(b, dim=1, keepdim=True)
    return 1. - torch.matmul(a, torch.transpose(b,0,1))

def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))

def _nn_euclidean_distance_torch(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    # x = x/((x*x).sum(1, keepdim = True)).sqrt()
    # y = y/((y*y).sum(1, keepdim = True)).sqrt()
    sim = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2).sqrt()
    # sim = sim.exp()
    # sim = (sim - 1)/(sim + 1)
    sim = torch.min(sim, 0)[0]
    return sim
    
def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

def _nn_cosine_distance_torch(x,y):
    '''
    Same as _nn_cosine_distance except torched
    '''
    distances = _cosine_distance_torch(x,y)
    return torch.min(distances, 0)[0]

class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
            self._metric_torch = _nn_euclidean_distance_torch
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
            self._metric_torch = _nn_cosine_distance_torch
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.budget = budget
        self.samples = {}
        self.samples_2d = {}

    def partial_fit(self, features, features_2d, targets, targets_2d, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for feature, target in zip(features, targets):
            if feature is not None:
                self.samples.setdefault(target, []).append(feature)
            else:
                self.samples.setdefault(target, [])
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets if k in targets}
        for target in active_targets:
            self.samples.setdefault(target, [])
        
        for feature_2d, target in zip(features_2d, targets_2d):
            self.samples_2d.setdefault(target, []).append(feature_2d)
            if self.budget is not None:
                self.samples_2d[target] = self.samples_2d[target][-self.budget:]

        self.samples_2d = {k: self.samples_2d[k] for k in active_targets}

    def distance(self, features, targets, compare_2d=False):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            if compare_2d:            
                cost_matrix[i, :] = self._metric(self.samples_2d[target], features)
            else:
                cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix

    def distance_torch(self, features, targets, compare_2d=False):
        '''
        Same as distance except torched.
        '''
        # features = torch.from_numpy(features).cuda()
        cost_matrix = torch.zeros(len(targets), len(features)).to('cuda:0')
        for i, target in enumerate(targets):
            if compare_2d:
                cost_matrix[i, :] = self._metric_torch(torch.stack(self.samples_2d[target], dim=0), features)
            else:
                cost_matrix[i, :] = self._metric_torch(torch.stack(self.samples[target], dim=0), features)
        return cost_matrix.cpu().numpy()

    def check_samples(self, targets):
        for target in targets:
            if len(self.samples[target]) == 0:
                return True
        return False
