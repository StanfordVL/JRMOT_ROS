import numpy as np
import pdb
from deep_sort_utils import non_max_suppression as deepsort_nms

def read_ground_truth_2d_detections(detection_path_2d, frame_idx, detection_matrix = None, threshold = -np.inf, nms_threshold = 0.75):
    if detection_matrix is None:
        detection_matrix = np.loadtxt(detection_path_2d, delimiter=',')

    if len(detection_matrix) == 0:
        return [], [], [], []
    if len(detection_matrix.shape) == 1:
        detection_matrix = np.expand_dims(detection_matrix, axis=0)

    frame_indices = detection_matrix[:, 0].astype(np.int32)
    if frame_idx is not None:
        mask = frame_indices == frame_idx
        detection_file = detection_matrix[mask]
    else:
        detection_file = detection_matrix

    frame_indices = detection_matrix[:, 0].astype(np.int32)
    if frame_idx is not None:
        conf = np.expand_dims(detection_file[:,6].astype(np.float32), 1)
        mask = conf[:,0] > threshold
        detection_file = detection_file[mask]
        object_ids = np.expand_dims(detection_file[:,1].astype(np.float32), 1)
        x1 = np.expand_dims(detection_file[:,2].astype(np.float32), 1)
        y1 = np.expand_dims(detection_file[:,3].astype(np.float32), 1)
        w = np.expand_dims(detection_file[:,4].astype(np.float32), 1)
        h = np.expand_dims(detection_file[:,5].astype(np.float32), 1)
        conf = np.expand_dims(detection_file[:,6].astype(np.float32), 1)
        cls_conf = -np.ones(conf.shape)
        cls_pred = -np.ones(conf.shape)
        detections = np.hstack([x1,y1,x1+w,y1+h, conf, cls_conf, cls_pred])
        boxes = np.hstack([x1, y1, w, h])
        indices = deepsort_nms(boxes, nms_threshold, np.squeeze(conf))
        detections_out = []
        for i in range(len(boxes)):
            if i in indices:
                detections_out.append(detections[i])
        if detections_out:
            detections = np.vstack(detections_out)
        else:
            detections = []
        return detections, object_ids, detection_matrix
    else:
        detections = []
        total_ids = []
        object_ids = np.expand_dims(detection_file[:,1].astype(np.float32), 1)
        for frame in np.unique(frame_indices):
            frame_mask = frame_indices==frame
            x1 = np.expand_dims(detection_file[frame_mask,2].astype(np.float32), 1)
            y1 = np.expand_dims(detection_file[frame_mask,3].astype(np.float32), 1)
            w = np.expand_dims(detection_file[frame_mask,4].astype(np.float32), 1)
            h = np.expand_dims(detection_file[frame_mask,5].astype(np.float32), 1)
            conf = np.expand_dims(detection_file[frame_mask,6].astype(np.float32), 1)
            boxes = np.hstack([x1, y1, w, h])
            cls_conf = -np.ones(conf.shape)
            cls_pred = -np.ones(conf.shape)
            frame_detections = np.hstack([x1,y1,x1+w,y1+h, conf, cls_conf, cls_pred])
            indices = deepsort_nms(boxes, nms_threshold, np.squeeze(conf))
            frame_detections_out = []
            ids = np.zeros((x1.shape[0], 1))
            for i in range(len(object_ids)):
                if i in indices:
                    frame_detections_out.append(frame_detections[i])
                elif i < ids.shape[0]:
                    ids[i] = -1
            if frame_detections_out:
                frame_detections = np.vstack(frame_detections_out)
                detections.append(frame_detections)
            total_ids.append(ids)

    detections = np.vstack(detections)
    ids = np.vstack(total_ids)

    frame_indices = frame_indices[np.squeeze(ids != -1)]
    object_ids = object_ids[np.squeeze(ids != -1)]

    return detections, object_ids, frame_indices

def read_ground_truth_3d_detections(detection_path_3d, frame_idx):
    
    detection_file = np.loadtxt(detection_path_3d, delimiter=',')
    frame_indices = detection_file[:, 0].astype(np.int32)
    if frame_idx is not None:
        mask = frame_indices == frame_idx
        detection_file = detection_file[mask]

    x = np.expand_dims(detection_file[:,2].astype(np.float32), 1)
    y = np.expand_dims(detection_file[:,3].astype(np.float32), 1)
    z = np.expand_dims(detection_file[:,4].astype(np.float32), 1)
    l = np.expand_dims(detection_file[:,5].astype(np.float32), 1)
    h = np.expand_dims(detection_file[:,6].astype(np.float32), 1)
    w = np.expand_dims(detection_file[:,7].astype(np.float32), 1)
    theta = np.expand_dims(detection_file[:,8].astype(np.float32), 1)
    ids = np.expand_dims(detection_file[:,1].astype(np.float32), 1)

    boxes_3d = np.hstack([x, y, z, l, h, w, theta])
    if frame_idx is None:
        return boxes_3d, ids, frame_indices
    return boxes_3d, ids
