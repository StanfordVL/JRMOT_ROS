import numpy as np
import os
import pdb
from tqdm import tqdm
from deep_sort_utils import non_max_suppression as deepsort_nms
from visualise import draw_track
import matplotlib.pyplot as plt
from PIL import Image


def evaluate_detections(detection_path_1, detection_path_2, detection_path_3, detection_path_4, gt_path):
	#expecting detections and gt in file with format as in read_detections.py
	# applies confidence thresholding
	try:
		detections_1 = np.loadtxt(detection_path_1, delimiter=',')
		# detections_2 = np.loadtxt(detection_path_2, delimiter=',')
		# detections_3 = np.loadtxt(detection_path_3, delimiter=',')
		# detections_4 = np.loadtxt(detection_path_4, delimiter=',')
		# detections = np.concatenate([detections_1, detections_2, detections_3, detections_4])
		detections = detections_1
		gt = np.loadtxt(gt_path, delimiter=',')
	except:
		return
	gt_frames = gt[:, 0]
	det_confidence = detections[:, 6]

	###CONFIDENCE THRESHOLD
	detections = detections[det_confidence > 0.9]
	########

	print("Average number of detections per frame = %f"%(detections.shape[0]/len(np.unique(gt_frames))))

	det_frames = detections[:, 0]
	det_confidence = detections[:, 6]
	gt_boxes = np.asarray(list(zip(gt[:, 2], gt[:, 3], gt[:, 4], gt[:, 5])))
	det_boxes = np.asarray(list(zip(detections[:, 2], detections[:, 3], detections[:, 4], detections[:, 5])))
	assignments = []
	missed_detections = 0
	for frame in np.unique(gt_frames):
		frame_mask_det = det_frames == frame
		frame_mask_gt = gt_frames == frame
		frame_gt_boxes = gt_boxes[frame_mask_gt]
		frame_det_boxes = det_boxes[frame_mask_det]
		frame_confidence = det_confidence[frame_mask_det]
		x1 = np.expand_dims(detections[frame_mask_det,2].astype(np.float32), 1)
		y1 = np.expand_dims(detections[frame_mask_det,3].astype(np.float32), 1)
		w = np.expand_dims(detections[frame_mask_det,4].astype(np.float32), 1)
		h = np.expand_dims(detections[frame_mask_det,5].astype(np.float32), 1)
		conf = np.expand_dims(detections[frame_mask_det,6].astype(np.float32), 1)
		boxes = np.hstack([x1, y1, w, h])
		indices = deepsort_nms(boxes, 0.75, np.squeeze(conf))
		frame_det_boxes = frame_det_boxes[indices]

		# print(frame_confidence)
		positive_arr = np.asarray([False]*len(frame_det_boxes))
		for i, gt_box in enumerate(frame_gt_boxes):
			iou_list = np.asarray([iou(gt_box, det_box) for det_box in frame_det_boxes])
			positive_idx = np.where(iou_list >= 0.5)[0]
			if len(positive_idx) == 0:
				missed_detections += 1
				plt.figure(0)

				plt.imshow(Image.open(os.path.join(os.path.split(detection_path_1)[0], '..','imgs','%.6d.png'%frame)))
				draw_track(None, gt_box, det = False)
				for det_box in frame_det_boxes:
					draw_track(None, det_box, det = True)
				# 	print(det_box)
				# print('Boxes:')
				# print(boxes)
				# print('FRAME DONE')
				plt.show()


			positive_arr[positive_idx] = True
		assignments.extend(list(zip(positive_arr, frame_confidence)))
	assignments = sorted(assignments, key = lambda x: x[1], reverse = True)
	predictions = list(zip(*assignments))[0]
	true_positives = np.cumsum(predictions)
	false_negatives = np.cumsum(predictions[::-1])[::-1]+missed_detections
	precision = true_positives/range(1,len(true_positives)+1)
	recall = true_positives/(true_positives + false_negatives)
	print("Total missed detections = %d"%missed_detections)
	base = 0
	idx = []
	for i,recall_val in enumerate(recall):
		if recall_val > base:
			base += 0.1
			idx.append(i)
		if base >1:
			break
	precision_vals = [np.amax(precision[index:]) for index in idx]
	if len(precision_vals) < 11:
		precision_vals.extend([0]*(11-len(precision_vals)))
	print(precision_vals)

	return np.mean(precision_vals)


def iou(bbox_1, bbox_2):

	x1_1, y1_1, w_1, h_1 = bbox_1
	x1_2, y1_2, w_2, h_2 = bbox_2
	x2_1 = x1_1 + w_1
	y2_1 = y1_1 + h_1

	x2_2 = x1_2 + w_2
	y2_2 = y1_2 + h_2
	
	area_1 = abs(x2_1 - x1_1)*abs(y2_1-y1_1)
	area_2 = abs(x2_2 - x1_2)*abs(y2_2-y1_2)


	intersection = max(0, (min(x2_1, x2_2) - max(x1_1, x1_2))) * max(0, (min(y2_1, y2_2) - max(y1_1, y1_2)))
	union = area_1 + area_2 - intersection

	return intersection / union

if __name__=='__main__':
	ap = []
	KITTI_root = 'data/KITTI/sequences'
	for sequence in tqdm(range(21)):
		ap.append(evaluate_detections(os.path.join(KITTI_root, '%.4d'%sequence, 'det','subcnn_car_det.txt'), 
										os.path.join(KITTI_root, '%.4d'%sequence, 'det','rrc_car_det.txt'),
										os.path.join(KITTI_root, '%.4d'%sequence, 'det','lsvm_car_det.txt'),
										os.path.join(KITTI_root, '%.4d'%sequence, 'det','regionlets_car_det.txt'), 
										os.path.join(KITTI_root, '%.4d'%sequence, 'gt', 'gt_car.txt')))
	ap = [ap_val for ap_val in ap if ap_val is not None]
	print("FINAL AVERAGE PRECISION OVER ALL SEQUENCES IS: %f"%np.mean(ap))

