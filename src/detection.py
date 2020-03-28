# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, box_3d, confidence, appearance_feature, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        # Note that detections format is centre of 3D box and dimensions (not bottom face)
        self.box_3d = box_3d
        if box_3d is not None:
            self.box_3d[1] -= box_3d[4]/2
            self.box_3d = np.asarray(box_3d, dtype=np.float32)
        self.confidence = float(confidence)
        self.appearance_feature = appearance_feature
        if feature is not None:
            self.feature = feature
        else:
            self.feature = None


    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    def to_xywh(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret
    def get_3d_distance(self):
        if self.box_3d is not None:
            return np.sqrt(self.box_3d[0]**2 + self.box_3d[2]**2)