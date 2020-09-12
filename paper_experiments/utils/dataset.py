import glob
import os
import pdb
import random
import sys
from itertools import compress

import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm

from models.pointnet_model import PointNet
from .calibration import Calibration, OmniCalibration
from .read_detections import (read_ground_truth_2d_detections,
                              read_ground_truth_3d_detections)


class SequenceDataset(Dataset):
    def __init__(self, folder_path, point_cloud=False, cuda=False, omni=False):

        self.files = sorted(glob.glob('%s/imgs/*.*' % folder_path), key = lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.files = [file for file in self.files if is_image_file(file)]
        self.point_cloud = point_cloud
        self.seq_name = os.path.split(folder_path)[-1]
        self.omni = omni
        if point_cloud:
            if self.omni:
                calib_folder = os.path.join(folder_path, 'calib')
                self.calib = OmniCalibration(calib_folder)
            else:
                self.calib_file = os.path.join(folder_path, 'calib', self.seq_name+'.txt')
                self.calib = Calibration(self.calib_file)
            self.depth_files = sorted(glob.glob('%s/*.*' % os.path.join(folder_path, 'depth')))
            self.depth_files = [file for file in self.depth_files if file.split('.')[-1]=='bin']
        else:
            self.calib = None
        self.cuda = cuda

    def __getitem__(self, index):


        img_path = self.files[index % len(self.files)]
        if self.point_cloud:
            depth_path = self.depth_files[index % len(self.depth_files)]
        # Extract image
        img = np.array(Image.open(img_path))

        # Channels-first
        input_img = np.transpose(img, (2, 0, 1))/255
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        if self.cuda:
            input_img = input_img.cuda()
        frame_idx = int(os.path.basename(img_path)[:-4])
        if self.point_cloud:
            #velodyne coordinates and image coordinates are different.
            #velo_x = camera_z
            #velo_y = -camera_x
            #velo_z = -camera_y
            if self.omni:
                scan = np.load(depth_path)
            else:
                scan = np.fromfile(depth_path, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            scan[:, :3] = self.calib.project_velo_to_ref(scan[:, :3])
            return frame_idx, img_path, input_img, scan
        else:
            return frame_idx, img_path, input_img, -1

    def __len__(self):
        return len(self.files)

def is_image_file(file):
    IMG_FILE_FORMATS = ['jpg', 'png', 'tif', 'bmp', 'jpeg']
    if file.split('.')[-1] in IMG_FILE_FORMATS:
        return True
    else:
        return False

class TripletDataset(Dataset):
    
    def __init__(self, feature_path, num_negative_samples = 100, cuda = True, sequence = False, test = False):
        if test:
            feature_file = os.path.join(feature_path, 'test_features.npy')
        else:
            feature_file = os.path.join(feature_path, 'features.npy')
        
        feature_array = np.load(feature_file)

        # feature_array = feature_array[:500]
        # feature_array = np.vstack([feature_array[146], feature_array[148], feature_array[149],feature_array[32], feature_array[10], feature_array[7],feature_array[9],feature_array[31], feature_array[8]])
        # feature_array = np.vstack([feature_array[10], feature_array[11],feature_array[9],feature_array[8],feature_array[249],feature_array[247]])
        # self.ids = feature_array[:, 0]
        # if not test:
        #     feature_array = feature_array[self.ids < 5]
        self.ids = feature_array[:, 0].astype(np.float32).astype(np.int32)
        self.unique_ids = np.unique(self.ids)
        self.frames = feature_array[:, 2].astype(np.float32).astype(np.int32)
        self.features = feature_array[:, 3:].astype(np.float32)
        self.sequences = feature_array[:, 1].astype(np.float32).astype(np.int32)
        self.sequence = sequence
        if self.sequence:
            self.size = self.unique_ids.size
        else:
            self.size = self.ids.size
        self.num_negative_samples = num_negative_samples
        self.tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def __getitem__(self, index):
        
        if self.sequence:
            object_id = self.unique_ids[index]
            positive_ids = self.ids == object_id
            object_sequence = self.sequences[positive_ids][0]
            object_frames = self.frames[positive_ids]

            positive_sequence = self.features[positive_ids]
            positive_sequence = torch.Tensor(positive_sequence).type(self.tensor_type)
            negative_sequence = []
            for frame in object_frames[1:]:
                idx = np.logical_and(self.sequences==object_sequence, self.frames==frame)
                idx = np.logical_and(idx, self.ids!=object_id)
                if np.sum(idx)==0:
                    negative_sequence.append(None)
                else:
                    negative_sequence.append(torch.Tensor(self.features[idx]).type(self.tensor_type))
            negative_ids = np.random.choice(len(self.ids), size = self.num_negative_samples, replace = False)
            negative_ids = negative_ids[self.ids[negative_ids] != object_id]
            negative_features = self.features[negative_ids]
            negative_features = torch.Tensor(negative_features).type(self.tensor_type)
            
            return positive_sequence, negative_sequence, negative_features
        else:
            object_id = self.ids[index]
            anchor_feature = self.features[index]
            anchor_feature = torch.Tensor(anchor_feature).type(self.tensor_type)

            positive_ids = np.where(self.ids == object_id)[0]
            positive_feature = self.features[random.choice(positive_ids)]
            positive_feature = torch.Tensor(positive_feature).type(self.tensor_type)

            negative_ids = np.random.choice(len(self.ids), size = self.num_negative_samples, replace = False)
            negative_ids = negative_ids[self.ids[negative_ids] != object_id]
            negative_features = self.features[negative_ids]
            negative_features = torch.Tensor(negative_features).type(self.tensor_type)


            return anchor_feature, positive_feature, negative_features


    def __len__(self):
        return self.size

class STIPDataset(Dataset):
    def __init__(self, folder_path, img_size=416, point_cloud = False, pad = False):

        self.files = sorted(glob.glob('%s/imgs/*/*.*' % folder_path), key = lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.files = [file for file in self.files if is_image_file(file)]
        self.img_shape = (img_size, img_size)
        self.pad = pad
        self.seq_name = os.path.split(folder_path)[-1]

    def __getitem__(self, index):


        img_path = self.files[index % len(self.files)]

        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        if self.pad:
            img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
            # Resize and normalize
            img = resize(img, (*self.img_shape, 3), mode='reflect', anti_aliasing = True)
        # Channels-first
        input_img = np.transpose(img, (2, 0, 1))/255
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img, -1

    def __len__(self):
        return len(self.files)

def collate_fn(inputs):
    #ASSUMES BATCH SIZE IS ALWAYS 1
    return inputs[0]