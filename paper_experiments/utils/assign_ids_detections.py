import numpy as np
import os
import pdb
from tqdm import tqdm
from deep_sort_utils import non_max_suppression as deepsort_nms
from visualise import draw_track
import matplotlib.pyplot as plt
from PIL import Image
from evaluate_detections import iou

def assign_detection_id(detection_path, gt_path, conf_threshold = 0, iou_threshold = 0.5):
	#expecting detections and gt in file with format as in read_detections.py
	# applies confidence thresholding
	try:
		detections = np.loadtxt(detection_path, delimiter=',')
		gt = np.loadtxt(gt_path, delimiter=',')
	except:
		return
	gt_frames = gt[:, 0]
	det_confidence = detections[:, 6]

	###CONFIDENCE THRESHOLD
	detections = detections[det_confidence > conf_threshold]
	########

	det_frames = detections[:, 0]
	det_confidence = detections[:, 6]
	gt_boxes = np.asarray(list(zip(gt[:, 2], gt[:, 3], gt[:, 4], gt[:, 5])))
	det_boxes = np.asarray(list(zip(detections[:, 2], detections[:, 3], detections[:, 4], detections[:, 5])))
	out_matrix = []
	assigned_ids = []

	for frame in np.unique(det_frames):
		frame_mask_det = det_frames == frame
		frame_mask_gt = gt_frames == frame
		gt_ids = gt[frame_mask_gt, 1]
		frame_gt_boxes = gt_boxes[frame_mask_gt]
		frame_det_boxes = det_boxes[frame_mask_det]

		for i, det_box in enumerate(frame_det_boxes):
			iou_list = np.asarray([iou(gt_box, det_box) for gt_box in frame_gt_boxes])
			iou_sorted = np.argsort(iou_list)
			positive_idx = np.where(iou_list >= iou_threshold)[0]
			if len(positive_idx)==0:
				assigned_ids.append(-1)
			else:
				assigned_ids.append(gt_ids[iou_sorted[-1]])
	assigned_ids = np.expand_dims(np.asarray(assigned_ids), 1)
	
	try:
		out_matrix = np.hstack([np.expand_dims(detections[:,0], 1), assigned_ids, detections[:,2:]])
	except:
		pdb.set_trace()
	np.savetxt(detection_path, out_matrix, delimiter=',', fmt = '%.2f')


	return

if __name__=='__main__':
	ap = []
	KITTI_root = 'data/KITTI/sequences'
	for sequence in tqdm(range(21)):
		assign_detection_id(os.path.join(KITTI_root, '%.4d'%sequence, 'det','rrc_subcnn_car_det.txt'), 
										os.path.join(KITTI_root, '%.4d'%sequence, 'gt', 'gt_car.txt'))
