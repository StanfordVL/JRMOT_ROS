import os
from os import listdir
from os.path import isfile, join

#root = "/cvgl2/u/mihirp/depth_tracking/data/JRDB/sequences/"
#root = "/cvgl2/u/mihirp/depth_tracking/data/JRDB/test_sequences/"
#root = "/cvgl2/u/mihirp/depth_tracking/data/KITTI/sequences/"
root = "/cvgl2/u/mihirp/depth_tracking/data/KITTI/test_sequences/"

file_name = "new_subcnn_faster_rcnn"
# file_name = "detectron2_x101"

def threshold(filename, thresh, min, max):
  detections = []
  with open(filename, 'r') as readfile:
    dets = readfile.read().split('\n')
    dets = dets[:len(dets)-1] #filter out last line which is just \n
    for det in dets:
      parsedet = det.split(' ')
      score = float(parsedet[len(parsedet)-1])
      parsedet[len(parsedet)-1] = str((float(parsedet[len(parsedet)-1]) - thresh) / (max - thresh))
      if(score > thresh):
        detections.append(parsedet)
  return detections

for seq in sorted(os.listdir(root)): #21 for normal, 29 for testing
  path = os.path.join(root,seq,'det')

  with open(os.path.join(path,file_name+'_raw.txt'), 'w') as f:
    pred_dets = []
    #pred_dets.append(threshold(os.path.join(path,'rrc.txt'), .05, 0, 1))
    pred_dets.append(threshold(os.path.join(path,'subcnn.txt'), .8, 0, 1))
    #pred_dets.append(threshold(os.path.join(path,'faster_rcnn.txt'), .99, 0, 1))
    #pred_dets.append(threshold(os.path.join(path,'detectron2_x101.txt'), .9, 0, 1))
    #pred_dets.append(threshold(path+'regionlets.txt', 5, -5, 25))
    if len(pred_dets[0]) == 0:
      continue
    max_frames = int((pred_dets[0])[len(pred_dets[0])-1][0])

    det_ctrs = [0,0,0,0]
    for frame in range(max_frames+1):
      frame_num = 0
      for j in range(1): #TODO: Update to number of detectors used
        while det_ctrs[j] < len(pred_dets[j]) and int( (pred_dets[j])[det_ctrs[j]][0]) == frame:
          (pred_dets[j])[det_ctrs[j]][1] = str(frame_num)
          frame_num+=1
          f.write( " ".join( (pred_dets[j])[det_ctrs[j]] )+'\n')
          det_ctrs[j]+=1

  # Counts max/min of scores
  for ctr, pred_det in enumerate(pred_dets):
    minval = 1000
    maxval = 0
    for detection in pred_det:
      score = detection[len(detection)-1]
      if float(score)>maxval:
        maxval = float(score)
      if float(score)<minval:
        minval = float(score)
    # print("Detector: "+str(ctr)+" Max: "+str(maxval))
    # print("Detector: "+str(ctr)+" Min: "+str(minval))

  with open(os.path.join(path,file_name+'_raw.txt'), 'r') as f:
    lines = f.readlines()

  with open(os.path.join(path, file_name+'_car.txt'), 'w') as fcar:
    with open(os.path.join(path, file_name+'_ped.txt'), 'w') as fped:
      for line in lines:
        if len(line) < 5:
          continue
        vals = line.split(' ')
        min_x = float(vals[6])
        min_y = float(vals[7])
        max_x = float(vals[8])
        max_y = float(vals[9])
        score = vals[-1]
        out_line = vals[0]+',0,'+str(min_x)+','+str(min_y)+','+str(max_x-min_x)+','+str(max_y-min_y)+','+str(score)
        if vals[2] == 'Car':
          fcar.write(out_line)
        elif vals[2] == 'Pedestrian':
          fped.write(out_line)
