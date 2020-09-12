import sys
sys.path.insert(0, '..')
import kalman_filter
import kf_simple3d
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
import pdb
np.set_printoptions(precision=4)

class Track:
    def __init__(self, track_id, first_detection, kf_type):
        # initiate kf
        if kf_type == "2d":
            self.kf = kalman_filter.KalmanFilter()
        elif kf_type == "simple3d":
            self.kf = kf_simple3d.KalmanFilterSimple3D()

        self.mean, self.cov = self.kf.initiate(first_detection)

        self.id = track_id
        n = len(self.mean)
        self.n = n
        m = len(first_detection)
        self.m = m

        # initialize data stores
        self.frame_log = np.zeros((0))
        self.measurement_log = np.zeros((0, m))
        self.gt_log = np.zeros((0, m))
        self.mean_log = np.zeros((0, n))
        self.cov_log = np.zeros((0, n, n))
        self.gating_distance_log = np.zeros((0))

    def update(self, measurement, gt, frame):

        # log data
        self.mean_log = np.vstack((self.mean_log, self.mean))
        self.cov_log = np.concatenate((self.cov_log, self.cov[np.newaxis,:,:]))
        self.measurement_log = np.vstack((self.measurement_log, measurement))
        self.gt_log = np.vstack((self.gt_log, gt))
        self.frame_log = np.append(self.frame_log, frame)

        gating_distance = self.kf.gating_distance(self.mean, self.cov, measurement)
        self.gating_distance_log = np.append(self.gating_distance_log, gating_distance)

        # KF predict and update
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, measurement)


    def plot(self):
        t = self.frame_log
        gt = self.gt_log
        meas = self.measurement_log
        state = self.mean_log

        plt.subplot(321)
        plt.plot(t, gt[:,0], label='Ground Truth')
        plt.plot(t, meas[:,0], label='Measured')
        plt.plot(t, state[:,0], label='filtered')
        plt.xlabel('time')
        plt.ylabel('x')
        plt.legend()

        plt.subplot(322)
        plt.plot(t, gt[:,1], label='Ground Truth')
        plt.plot(t, meas[:,1], label='Measured')
        plt.plot(t, state[:,1], label='filtered')
        plt.xlabel('time')
        plt.ylabel('y')
        plt.legend()

        plt.subplot(323)
        plt.plot(gt[:,0], gt[:,1], label='Ground Truth')
        plt.plot(meas[:,0], meas[:,1], label='Measured')
        plt.plot(state[:,0], state[:,1], label='filtered')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()


        plt.subplot(324)
        plt.plot(t, state[:,self.m], label='filtered')
        plt.xlabel('time')
        plt.ylabel('Vx')
        plt.legend()

        plt.subplot(325)
        plt.plot(t, state[:,self.m+1], label='filtered')
        plt.xlabel('time')
        plt.ylabel('Vy')
        plt.legend()

        plt.show()


def file2data(fname):
    # data should be a list of lists of numpy arrays
    # Each element in the list represents a frame
    # Each frame is a list of detections
    # Each detection is a numpy array of measurements. 
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data

def cmp_tracks(track1, track2):
    # Expect perfect match in mean log and gating distance
    mean_log_pass = np.max(np.abs(track1.mean_log == track2.mean_log)) > 1e-12
    gating_distance_pass = np.max(np.abs(track1.gating_distance_log == track2.gating_distance_log)) > 1e-12
    return mean_log_pass and gating_distance_pass

def cmp(data, val):
    any_fail = False
    for itrack in data:
        passed = cmp_tracks(data[itrack], val[itrack])
        if not passed:
            print("Mismatch found in track: ", itrack)
            # pdb.set_trace()
            any_fail = True
        else: 
            print("Tracks matched: ", itrack)

    return not any_fail


def validate(data, fname):
    if os.path.isfile(fname):
        val_data = file2data(fname)
        return cmp(data, val_data)
    else:
        with open(fname, "wb") as f:
            pickle.dump(data, f)   
        return True

def run_kf_test(fname, kf_type):
    print("Running test for: {}".format(fname))
    data = file2data(fname)

    first_frame = data[0]

    tracks = {}
    for detection in first_frame:
        meas, gt, gt_id = (detection[0], detection[1], detection[2])
        tracks[gt_id] = Track(gt_id, meas, kf_type)

    frame_cnt = 0; 
    for frame in data:
        for detection in frame:
            meas, gt, gt_id = (detection[0], detection[1], detection[2])
            tracks[gt_id].update(meas, gt, frame_cnt)

        frame_cnt += 1

    passed = validate(tracks, fname + ".val")

    if not passed:
        for track_id in tracks:
            tracks[track_id].plot()


if __name__=='__main__':
    run_kf_test("single_track_4state_test.p", "2d")
    run_kf_test("two_track_4state_test.p", "2d")
    run_kf_test("single_track_6state_test.p", "simple3d")



