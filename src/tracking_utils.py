import torch, sys, os, pdb
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from aligned_reid_utils import load_state_dict
from featurepointnet_model_util import rotate_pc_along_y
from deep_sort_utils import non_max_suppression as deepsort_nms
import math
from detection import Detection


def create_detector(config_path, weight_path, cuda):

    detector = Darknet(config_path)
    detector.load_weights(weight_path)
    if cuda:
        detector.cuda()
    detector.eval()
    return detector

def get_depth_patches(point_cloud, box_3d, ids_3d, rot_angles, num_point = 1024):
    #print(ids_3d)
    depth_patches = []
    for i, box in enumerate(box_3d):
        if ids_3d[i] == -1:
            depth_patches.append(None)
            continue
        box_center = np.asarray([ [box[0], box[1], box[2]] ])
        rotate_pc_along_y(box_center, np.pi/2 + np.squeeze(box[6]))
        box_center = box_center[0]
        rotate_pc_along_y(point_cloud, np.pi/2 + np.squeeze(box[6]))
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        idx_1 = np.logical_and(x >= float(box_center[0] - box[3]/2.0), x <= float(box_center[0] + box[3]/2.0))
        idx_2 = np.logical_and(y <= (box_center[1]+0.1), y >= float(box_center[1] - box[4]))
        idx_3 = np.logical_and(z >= float(box_center[2] - box[5]/2.0), z <= float(box_center[2] + box[5]/2.0))
        idx = np.logical_and(idx_1, idx_2)
        idx = np.logical_and(idx, idx_3)
        depth_patch = point_cloud[idx, :]
        rotate_pc_along_y(point_cloud, -(np.squeeze(box[6])+np.pi/2)) #unrotate to prep for next iteration
        rotate_pc_along_y(depth_patch, -(np.squeeze(box[6])+np.pi/2))

        if depth_patch.size == 0:
            ids_3d[i] = -1
            depth_patches.append(None)
        else:
            if depth_patch.shape[0] > num_point:
                pc_in_box_fov = np.expand_dims(depth_patch[np.random.choice(range(depth_patch.shape[0]), size = (num_point), replace=False)], 0)
            else:

                pc_in_box_fov = np.expand_dims(
                            np.vstack([depth_patch,
                            depth_patch[np.random.choice(range(depth_patch.shape[0]), size = (num_point - depth_patch.shape[0]), replace=True)]])
                            , 0)
            depth_patches.append( get_center_view_point_set(pc_in_box_fov, rot_angles[i])[0])

    return depth_patches, ids_3d


def non_max_suppression_3D_prime(detections, boxes_3d, ids_3d, ids_2d, nms_thresh = 1, confidence = None):
    x = [boxes_3d[i][0] for i in range(len(boxes_3d))]
    z = [boxes_3d[i][2] for i in range(len(boxes_3d))]
    l = [boxes_3d[i][5] for i in range(len(boxes_3d))] #[3]
    w = [boxes_3d[i][3] for i in range(len(boxes_3d))] #[5]
    indices = deepsort_nms(boxes_3d, nms_thresh, np.squeeze(confidence))
    for i in range(len(ids_3d)):
        if i not in indices:
            ids_3d[i] = -1
            ids_2d[i] = -1
            boxes_3d[i] = None
            detections[i] = None
    return detections, boxes_3d, ids_2d, ids_3d

def non_max_suppression_3D(depth_patches, ids_3d, ids_2d, nms_thresh = 1, confidence = None):
    #depth_patches list of patches

    if len(depth_patches) == 0:
        return []

    pick = []

    if confidence is not None:
        idxs = np.argsort(confidence)
    else:
        idxs = list(range(len(depth_patches)))

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        overlap = np.asarray([iou_3d(depth_patches[i], depth_patches[idxs[x]]) for x in range(last)])
        if np.any(overlap == -np.inf):
            idxs = np.delete(idxs, [last])
            continue
        pick.append(i)
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > nms_thresh)[0])))
    for i in range(len(depth_patches)):
        if i not in pick:
            if ids_3d[i]!=-1:
                ids_2d[i] = -1
            ids_3d[i] = -1
    return depth_patches, ids_3d, ids_2d

def iou_3d(patch_1, patch_2):
    #Expecting patches of shape (N, 4) or (N,3) (numpy arrays)
    if patch_2 is None:
        return np.inf
    elif patch_1 is None:
        return -np.inf
    # Unique points
    patch_unique_1 = np.unique(patch_1, axis = 0)
    patch_unique_2 = np.unique(patch_2, axis = 0)
    intersection_points = 0
    for point_1_idx in range(patch_unique_1.shape[0]):
        point_distance = np.sqrt(np.sum((patch_unique_1[point_1_idx]-patch_unique_2)**2, axis = 1))
        intersection_points += np.any(point_distance<0.3)

    union_points = patch_unique_1.shape[0] + patch_unique_2.shape[0] - intersection_points

    iou = intersection_points/union_points

    return iou

def convert_detections(detections, features, appearance_features, detections_3d):
    detection_list = []
    if detections_3d is None:
        detections_3d = [None] * len(detections)
    for detection, feature, appearance_feature, detection_3d in zip(detections, features, appearance_features, detections_3d):
        x1, y1, x2, y2, conf, _, _ = detection
        box_2d = [x1, y1, x2-x1, y2-y1]
        if detection_3d is not None:
            x, y, z, l, w, h, theta = detection_3d
            box_3d = [x, y, z, l, w, h, theta]
        else:
            box_3d = None
        if feature is None:
            detection_list.append(Detection(box_2d, None, conf, appearance_feature, feature))
        else:
            detection_list.append(Detection(box_2d, box_3d, conf, appearance_feature, feature))

    return detection_list

def combine_features(features, depth_features, ids_3d, combination_model, depth_weight=1):

    combined_features = []
    appearance_features = []
    for i, (appearance_feature, depth_feature) in enumerate(zip(features, depth_features)):
        if not ids_3d[i]:
            depth_feature = torch.zeros(512, device=torch.device("cuda:0"))
        # appearance_feature = torch.zeros(512, device=torch.device("cuda:0"))
        combined_features.append(torch.cat([appearance_feature, depth_feature* depth_weight]))
        appearance_features.append(appearance_feature)

    if combination_model is not None and len(combined_features) > 0:
        combination_model.eval()
        combined_feature = torch.stack(combined_features)
        combined_features = combination_model(combined_feature).detach()
        combined_features = list(torch.unbind(combined_features))
    return combined_features, appearance_features

def filter(detections):
    for i, det in enumerate(detections): #Note image is 1242 x 375
        left = det[0]
        top = det[1]
        right = det[2]
        bottom = det[3]
        if (left < 10 or right > 1232) and (top < 10 or bottom > 365):
            detections[i] = None
    return detections
