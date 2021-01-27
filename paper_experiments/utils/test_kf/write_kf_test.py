import sys
sys.path.insert(0, '..')
import kalman_filter
import numpy as np
import pickle
import pdb

def data2file(data, fname):
    # data should be a list
    # Each element in the list represents a frame
    # Each frame is a list of detections
    # Each detection is a list of [measurement, ground truth, gt track id]

    with open(fname, "wb") as f:
        pickle.dump(data, f)   

def add_noise(center, std):
    out = np.zeros_like(center)
    for i in range(len(center)):
        out[i] = np.random.normal(center[i], std[i])
    return out

def single_track_4state_test():
    np.random.seed(0)

    data = []

    # Iterate over frames
    for i in range(100):
        frame = []
        track_id = 0

        # Track 0
        x_gt = i*10
        y_gt = i*10
        a_gt = 2
        h_gt = 400

        gt = np.array([x_gt, y_gt, a_gt, h_gt])
        meas = add_noise(gt, [15, 15, 0.1, 5])

        detection = [meas, gt, 0]
        frame.append(detection)

        data.append(frame)

    data2file(data, "single_track_4state_test.p")

def two_track_4state_test():
    np.random.seed(0)

    data = []

    # Iterate over frames
    for i in range(100):
        frame = []
        track_id = 0

        # Track 0
        x_gt = i*10
        y_gt = i*10
        a_gt = 2
        h_gt = 400

        gt = np.array([x_gt, y_gt, a_gt, h_gt])
        meas = add_noise(gt, [15, 15, 0.1, 5])

        detection = [meas, gt, track_id]
        frame.append(detection)

        # Track 1
        track_id += 1
        x_gt = i*i/10
        y_gt = i**0.5*30
        a_gt = 2
        h_gt = 400

        gt = np.array([x_gt, y_gt, a_gt, h_gt])
        meas = add_noise(gt, [20, 10, 0.2, 8])

        detection = [meas, gt, track_id]
        frame.append(detection)

        data.append(frame)

    data2file(data, "two_track_4state_test.p")

def single_track_6state_test():
    np.random.seed(0)

    data = []

    # Iterate over frames
    for i in range(100):
        frame = []
        track_id = 0

        # Track 0
        x_gt = i*10
        y_gt = i*10
        l_gt = 400
        h_gt = 400
        w_gt = 400
        theta_gt = i/10

        gt = np.array([x_gt, y_gt, l_gt, h_gt, w_gt, theta_gt])
        gt = add_noise(gt, [3, 3, 1, 1, 1, 1*0.017])
        meas = add_noise(gt, [15, 15, 5, 5, 5, 5*0.017])

        detection = [meas, gt, 0]
        frame.append(detection)

        data.append(frame)

    data2file(data, "single_track_6state_test.p")

if __name__=='__main__':
    single_track_4state_test()
    two_track_4state_test()
    single_track_6state_test()