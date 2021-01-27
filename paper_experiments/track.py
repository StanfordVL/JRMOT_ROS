import open3d as o3d
import torch
import argparse
import os, pdb, sys, copy, pickle
import time
import random
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.aligned_reid_model import Model as aligned_reid_model
from utils.yolo_utils.utils import non_max_suppression, load_classes
from models.combination_model import CombiNet, CombiLSTM
from utils.dataset import SequenceDataset, STIPDataset, collate_fn
from models.deep_sort_model import ImageEncoder as deep_sort_model
from utils.tracker import Tracker
from utils.tracker_3d import Tracker_3d
from utils.deep_sort_utils import non_max_suppression as deepsort_nms
from utils.visualise import draw_track
from utils.read_detections import read_ground_truth_2d_detections, read_ground_truth_3d_detections
from utils.tracking_utils import create_detector, convert_detections, combine_features
from utils.tracking_utils import non_max_suppression_3D, non_max_suppression_3D_prime
from utils.aligned_reid_utils import generate_features, generate_features_batched, get_image_patches, create_appearance_model
from utils.featurepointnet_model_util import generate_detections_3d, convert_depth_features
from models.featurepointnet_model import create_depth_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_folder', type=str, default='data/KITTI/sequences/0001', help='path to image sequence')
    parser.add_argument('--output_folder', type=str, default='results', help='output folder')
    parser.add_argument('--aligned_reid_ckpt', type=str, default='weights/aligned_reid_market_weights.ckpt', help='path to model config file')
    parser.add_argument('--resnet_reid_ckpt', type=str, default='weights/resnet_reid.ckpt', help='path to model config file')
    parser.add_argument('--depth_model', type=str, default='FPointNet', help='type of depth model to use')
    parser.add_argument('--depth_config_path', type=str, default='config/featurepointnet.cfg', help='path to model config file')
    parser.add_argument('--appearance_model', type=str, default='resnet_reid', help='type of appearance model to use aligned_reid or deepsort or resnet_reid')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--depth_weight', type=float, default=1, help='weight of depth feature while concatenating')
    parser.add_argument('--nms_thresh', type=float, default=0.56, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
    parser.add_argument('-p', '--point_cloud', action='store_false', help='Use to disable pointcloud')
    parser.add_argument('-o', '--optical_flow_initiation', action='store_false', help='Use to enable optical flow based velocity initiation')
    parser.add_argument('-q', '--perfect', action='store_true', help='whether to use perfect assignments')
    parser.add_argument('-g', '--ground_truth', action='store_true', help='whether to use ground truth detections')
    parser.add_argument('-r', '--reference', action='store_false', help='whether to use reference detections')
    parser.add_argument('-t', '--track_3d', action='store_true', help='whether to do 3d tracking')
    parser.add_argument('--ref_det', type = str, default = 'new_rrc_subcnn_car', help='lsvm, subcnn, regionlets, maskrcnn')
    parser.add_argument("--nn_budget", type=int, default=100, help="Maximum size of the appearance descriptors gallery. If None, no budget is enforced.")
    parser.add_argument("--dummy_node_cost_app", type=float, default=0.99, help="Dummy node appearance cost for JPDA (or maximum distnce when using deepsort)")
    parser.add_argument("--dummy_node_cost_iou", type=float, default=0.97, help="Dummy node iou cost for JPDA (or maximum distnce when using deepsort)")
    parser.add_argument("-c", "--combine_features", action = 'store_false', help="Whether to use trained MLP to combine features")
    parser.add_argument("-f", "--fpointnet", action = 'store_false', help="Whether to use F-PointNet for 3d detection")
    parser.add_argument("--combo_model", default = 'weights/resnet_reid_fpointnet_combo_car/mlp__1570759353.0113978/best_checkpoint.tar"', help="Trained MLP checkpoint to combine features")
    parser.add_argument("-j", "--JPDA", action = 'store_false', help="Whether to use JPDA for soft assignments")
    parser.add_argument("-l", "--LSTM", action = 'store_true', help="Whether to use LSTM for feature combination and update")
    parser.add_argument("--lstm_model", default = 'weights/aligned_reid_fpointnet_combo/lstm/best_checkpoint.tar', help="Trained LSTM checkpoint to combine features")
    parser.add_argument("-m","--m_best_sol", type=int, default=10, help="Number of solutions for JPDA")
    parser.add_argument("--log_data", action='store_true', help="Turn on full data logging")
    parser.add_argument("--max_age", type=int, default=2, help="Number of misses before termination")
    parser.add_argument("--n_init", type=int, default=2, help="Consecutive frames for tentative->confirmed")
    parser.add_argument("--assn_thresh", type=float, default=0.65, help="min prob for match")
    parser.add_argument("--matching_strategy", type=str, default="hungarian", help="matching strategy for JPDA (max_and_threshold, strict_max_pair, or hungarian)")
    parser.add_argument("--kf_appearance_feature", type=bool, default=False, help="Whether to use kf state for apperance features")
    parser.add_argument('-i', "--use_imm", action = 'store_true', help='Whether to use IMM')
    parser.add_argument('-v', "--verbose", action = 'store_true', help='Verbose')
    parser.add_argument('--kf_process', type=float, default=5.2, help='kf 2d process noise factor')
    parser.add_argument('--kf_2d_meas', type=float, default=3.2, help='kf 2d measurement noise factor')
    parser.add_argument('--kf_3d_meas', type=float, default=0.25, help='kf 3d measurement noise factor')
    parser.add_argument('--pos_weight_3d', type=float, default=1, help='Weight on position covariance process noise in KF')
    parser.add_argument('--pos_weight', type=float, default=0.006, help='Weight on position covariance process noise in KF')
    parser.add_argument('--vel_weight', type=float, default=0.008, help='Weight on velocity covariance process noise in KF')
    parser.add_argument('--theta_weight', type=float, default=0.02, help='Weight on velocity covariance process noise in KF')
    parser.add_argument('--gate_limit', type=float, default=600, help='Maximum covariance value of the gate')
    parser.add_argument('--initial_uncertainty', type=float, default=1, help='Uncertainty scaling for initial covariance of track')
    parser.add_argument('--uncertainty_limit', type=float, default=1.5, help='Uncertainty limit at which to terminate tracks')
    parser.add_argument("--gate_full_state", action='store_true', help="Whether to gate on full kalman state, default is only position")
    parser.add_argument("--near_online", action = 'store_true', help="Whether to do near online tracking")
    parser.add_argument("--omni", action = 'store_true', help="Omni directional camera (JRDB)")
    opt = parser.parse_args()
    opt.sequence_folder = opt.sequence_folder.rstrip(os.sep)
    opt.using_cuda = torch.cuda.is_available() and opt.use_cuda
    if not opt.point_cloud and opt.track_3d:
        raise("Must provide point cloud if doing 3D tracking!")
    if opt.verbose:
        print(opt)
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)
    return opt

# @profile
def main(opt):

    if opt.verbose:
        print("------------------------")
        print("RUNNING SET UP")
        print("------------------------")
    tf.logging.set_verbosity(40)
    random.seed(0)
    Tensor = torch.cuda.FloatTensor if opt.using_cuda else torch.FloatTensor
    os.makedirs(opt.output_folder, exist_ok=True)
    if opt.LSTM:
        opt.max_cosine_distance = 1
        lstm = CombiLSTM()
        checkpoint = torch.load(opt.lstm_model)
        lstm.load_state_dict(checkpoint['state_dict'])
        if opt.using_cuda:
            lstm.cuda()
        lstm.eval()
    else:
        lstm = None
    if opt.combine_features:
        combination_model = CombiNet()
        checkpoint = torch.load(opt.combo_model)
        combination_model.load_state_dict(checkpoint['state_dict'])
        if opt.using_cuda:
            combination_model.cuda()
        combination_model.eval()
    else:
        combination_model = None
    
    dataset = SequenceDataset(opt.sequence_folder, point_cloud=opt.point_cloud, omni=opt.omni)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.n_cpu, collate_fn = collate_fn)
    appearance_model = create_appearance_model(opt.appearance_model, opt.aligned_reid_ckpt, opt.resnet_reid_ckpt, opt.using_cuda)
    if opt.point_cloud:
        depth_model = create_depth_model(opt.depth_model, opt.depth_config_path)
    if opt.track_3d:
        tracker = Tracker_3d(appearance_model=appearance_model, cuda=opt.using_cuda, JPDA = opt.JPDA, m_best_sol=opt.m_best_sol,
                        max_age = opt.max_age, n_init=opt.n_init, assn_thresh=opt.assn_thresh,
                        matching_strategy=opt.matching_strategy,
                        gate_full_state=opt.gate_full_state,
                        kf_vel_params=(opt.pos_weight_3d, opt.pos_weight, opt.vel_weight, opt.theta_weight,
                                       opt.kf_process, opt.kf_2d_meas, opt.kf_3d_meas, opt.initial_uncertainty),
                        calib=dataset.calib,
                        dummy_node_cost_iou=opt.dummy_node_cost_iou,
                        dummy_node_cost_app=opt.dummy_node_cost_app,
                        nn_budget=opt.nn_budget,
                        use_imm=opt.use_imm,
                        uncertainty_limit=opt.uncertainty_limit,
                        gate_limit=opt.gate_limit,
                        omni=opt.omni)
    else:
        tracker = Tracker(appearance_model=appearance_model, cuda=opt.using_cuda, JPDA = opt.JPDA, m_best_sol=opt.m_best_sol,
                        max_age = opt.max_age, n_init=opt.n_init, assn_thresh=opt.assn_thresh,
                        matching_strategy=opt.matching_strategy,
                        kf_appearance_feature=opt.kf_appearance_feature,
                        gate_full_state=opt.gate_full_state,
                        kf_vel_params=(opt.pos_weight, opt.vel_weight, opt.kf_process, opt.kf_2d_meas, opt.initial_uncertainty),
                        kf_walk_params=(opt.pos_weight, opt.vel_weight, opt.kf_process, opt.kf_2d_meas, opt.initial_uncertainty),
                        calib=dataset.calib,
                        dummy_node_cost_iou=opt.dummy_node_cost_iou,
                        dummy_node_cost_app=opt.dummy_node_cost_app,
                        nn_budget=opt.nn_budget,
                        use_imm=opt.use_imm,
                        uncertainty_limit=opt.uncertainty_limit,
                        optical_flow=opt.optical_flow_initiation,
                        gate_limit=opt.gate_limit)

    results = []
    results_3d = []
    n_frames = len(dataloader)
    if opt.log_data:
        full_log = [{'tracks':[], 'detections':[], 'detections_3d':[]} for _ in range(n_frames)]
    det_matrix = None
    seq_name = os.path.split(opt.sequence_folder)[-1]

    frame_times = []
    if opt.verbose:
        print("------------------------")
        print("BEGINNING TRACKING OF SEQUENCE %s"%seq_name)
        print("------------------------")
    for frame_idx, img_path, input_img, point_cloud in tqdm(dataloader, ncols = 100, disable=not opt.verbose):
        # if frame_idx > 120:
        #     break
        # elif frame_idx < 98:
        #     continue

        if opt.log_data:
            full_log[frame_idx]['img_path'] = copy.copy(img_path)
        input_img = input_img.type(Tensor)
        if opt.reference:
            detections, object_ids, det_matrix = read_ground_truth_2d_detections(os.path.join(opt.sequence_folder,'det',opt.ref_det+'.txt'), frame_idx, det_matrix, threshold = 0, nms_threshold = opt.nms_thresh)
        elif opt.ground_truth:
            detections, object_ids, det_matrix = read_ground_truth_2d_detections(os.path.join(opt.sequence_folder,'gt','gt.txt'), frame_idx, det_matrix, nms_threshold = opt.nms_thresh)
        else:
            raise("Must specify ground truth or detections")

        # --- START OF TRACKING ---
        # start_time = time.time()
        if detections is None or len(detections)==0:
            tracker.predict()
            if opt.log_data:
                full_log[frame_idx]['predicted_tracks'] = copy.deepcopy(tracker.tracks)
            start_time = time.time()
            tracker.update(input_img, [])
        else:
            total_dets = len(detections)
            patches = get_image_patches(input_img, detections)
            appearance_features = generate_features_batched(appearance_model, patches, opt, object_ids)
            if opt.point_cloud:
                if not opt.omni:
                    point_cloud = point_cloud[point_cloud[:,2]>=0]
                if opt.fpointnet:
                    boxes_3d, valid_3d, _, scores_3d, depth_features = generate_detections_3d(depth_model, 
                                                                        detections, np.asarray(point_cloud), 
                                                                        dataset.calib, input_img.shape,
                                                                        peds='ped' in opt.ref_det or opt.omni)
                    depth_features = convert_depth_features(depth_features, valid_3d)
                else:
                    boxes_3d, valid_3d = read_ground_truth_3d_detections(os.path.join(opt.sequence_folder,'gt','3d_detections.txt'), frame_idx)        
                features, appearance_features = combine_features(appearance_features, depth_features, valid_3d, combination_model, depth_weight = opt.depth_weight)
                # boxes_3d = boxes_3d[valid_3d != -1] # Old and buggy way of handling missing box
                # detections = detections[valid_3d != -1]
                if np.any(valid_3d == -1):
                    compare_2d = True
                else:
                    compare_2d = False
                if len(boxes_3d) > 0:
                    detections_3d = []
                    for idx, box in enumerate(boxes_3d):
                        if valid_3d[idx] == -1:
                            detections_3d.append(None)
                        else:
                            detections_3d.append(np.array(box).astype(np.float32))
                else:
                    detections_3d = None
            else:
                appearance_features = [appearance_features[i] for i in range(total_dets)]
                features = [None]*len(appearance_features)
                compare_2d = True
                detections_3d = None
            detections = convert_detections(detections, features, appearance_features, detections_3d)
            tracker.predict()
            if opt.log_data:
                full_log[frame_idx]['predicted_tracks'] = copy.deepcopy(tracker.tracks)
            start_time = time.time()
            tracker.update(input_img, detections, compare_2d)

        # --- END OF TRACKING ---
        end_time = time.time()
        frame_times.append(end_time - start_time)


        if opt.log_data:
            full_tracks = copy.deepcopy(tracker.tracks)
            temp_tracks = []
            for track in full_tracks:
                bbox = track.to_tlwh(None)
                if not (bbox[0] < 0-10 or bbox[1] < 0-10 or bbox[0] + bbox[2] > input_img.shape[2]+10 or bbox[1] + bbox[3] > input_img.shape[1]+10):
                    temp_tracks.append(track)
            full_log[frame_idx]['tracks'] = temp_tracks
            full_log[frame_idx]['detections'] = copy.deepcopy(detections)

        for track in tracker.tracks:
            if opt.track_3d:
                bbox_3d = track.to_tlwh3d()
            else:
                bbox = track.to_tlwh(None)
            if bbox[0] < 0-10 or bbox[1] < 0-10 or bbox[0] + bbox[2] > input_img.shape[2]+10 or bbox[1] + bbox[3] > input_img.shape[1]+10:
                continue
            bbox[0] = max(0,bbox[0]) # Frame adjustments
            bbox[1] = max(0,bbox[1])
            bbox[2] = min(bbox[0]+bbox[2], input_img.shape[2])-bbox[0]
            bbox[3] = min(bbox[1]+bbox[3], input_img.shape[1])-bbox[1]

            track_status = 1
            if not track.is_confirmed(): # or track.time_since_update > 0:
                if opt.near_online:
                    if not track.is_confirmed():
                         track_status = 0
                    else:
                         track_status = 2
                         continue
                else:
                    continue
            if opt.near_online:
                if opt.track_3d:
                    results_3d.append([frame_idx, track.track_id, bbox_3d[0], bbox_3d[1], bbox_3d[2], bbox_3d[3], bbox_3d[4], bbox_3d[5], bbox_3d[6], track_status])
                else:
                    results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], track_status])

                if track_status == 1: #updates 0s
                    for row_i in range(len(results)):
                        if results[row_i][1] == track.track_id:
                            results[row_i][6] = 1
                        if opt.point_cloud:
                            if results_3d[row_i][1] == track.track_id:
                                results_3d[row_i][7] = 1
            else:
                if opt.track_3d:
                    results_3d.append([frame_idx, track.track_id, bbox_3d[0], bbox_3d[1], bbox_3d[2], bbox_3d[3], bbox_3d[4], bbox_3d[5], bbox_3d[6]])
                else:
                    results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
                # if opt.point_cloud:

    frame_times = np.asarray(frame_times)
    if opt.verbose:
        print("------------------------")
        print("COMPLETED TRACKING, SAVING RESULTS")
        print("------------------------")
        print('\n\n','Total Tracking Time:',np.sum(frame_times),'Average Time Per Frame:',np.mean(frame_times))

    if opt.track_3d:
        output_file_3d = os.path.join(opt.output_folder, seq_name+"_3d.txt")
        if len(results_3d) > 0:
            with open(output_file_3d, 'w+') as f:
                for row in results_3d:
                    if opt.near_online and row[9] != 1:
                        continue
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,1,1,1,-1' % (
                        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]), file=f)
    else:
        output_file = os.path.join(opt.output_folder, seq_name+".txt")
        if len(results) > 0:
            with open(output_file, 'w+') as f:
                for row in results:
                    if opt.near_online and row[6] != 1:
                        continue
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1,-1' % (
                        row[0], row[1], row[2], row[3], row[4], row[5]), file=f)

    if opt.log_data:
        output_file = os.path.join(opt.output_folder, seq_name+".p")
        with open(output_file, 'wb') as f:
            pickle.dump(full_log, f)

if __name__=='__main__':
    opt = parse_arguments()
    main(opt)
