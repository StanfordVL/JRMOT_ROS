import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import numpy as np
import utils.imm as imm
from PIL import Image
import pdb

def draw_track(bbox, track = None, bbox_colors = None, det = True,
               do_ellipse = False, axis = None, id_num = 0, do_velocity=False):
    if axis is None:
        axis = plt.gca()
    if track is None:
        color = plt.get_cmap('tab20b')(8) if det else plt.get_cmap('tab20b')(6)
        # plt.imshow(original_img)
        width = bbox[2]
        height = bbox[3]
    else:
        color = bbox_colors[track.track_id]
        id_num = track.track_id
        width = bbox[2]
        height = bbox[3]

    plot_bbox = plt_patches.Rectangle((bbox[0], bbox[1]), width, height, linewidth=2,
                        edgecolor=color,
                        facecolor='none')
    ax = axis
    ax.add_patch(plot_bbox)
    ax.text(bbox[0], bbox[1], s = id_num, color='white', verticalalignment='top',
        bbox={'color': color, 'pad': 0})
  
    if do_ellipse:
        draw_ellipse(track, color)
    if do_velocity:
        draw_velocity(track, color)


def draw_detection(detection, color='k'):
    bbox = detection.tlwh
    plot_bbox = plt_patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2,
                        edgecolor=color,
                        facecolor = 'w',
                        alpha = 0.5)
    ax = plt.gca()
    ax.add_patch(plot_bbox)
   
def draw_ellipse(track, color):
    ax = plt.gca()
    if track.model_probabilities is not None:
        mean, cov = imm.IMMFilter2D.combine_states(track.mean, track.covariance, track.model_probabilities)
        # print("New orig mat",track.covariance)
        # print("New",cov)
    else:
        mean = track.mean
        cov = track.covariance
        # print("Old",cov)

    lambda_, v = np.linalg.eig(cov[:2, :2])
    lambda_ = np.sqrt(lambda_)
    idx = np.argsort(lambda_)[::-1]
    lambda_ = lambda_[idx]
    v = v[:, idx]
    nsigma = np.sqrt(5.99)
    ell = plt_patches.Ellipse(xy=(mean[0], mean[1])
                              , width= lambda_[0]*2*nsigma 
                              , height=lambda_[1]*2*nsigma
                              , angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))
                              , edgecolor=color
                              , facecolor='none'
                              )
    ax.add_patch(ell)

def draw_velocity(track, color):
    ax = plt.gca()
    if track.model_probabilities is not None:
        mean, cov = imm.IMMFilter2D.combine_states(track.mean, track.covariance, track.model_probabilities)
    else:
        mean = track.mean
    ax.arrow(mean[0], mean[1], 
            mean[4], mean[5],
            edgecolor=color,
            head_width=5)

def draw_box3d(mu, color, alpha, facecolor='none', ax=None):
    if np.any(np.isnan(mu)):
        return
    if ax is None:
        ax = plt.gca()
    x, z, l, w, theta = mu[0], mu[2], mu[3], mu[5], mu[6]
    r = np.sqrt(w**2 + l**2)/2
    psi = np.arctan2(w, l)
    dx, dz = r*np.cos(psi), r*np.sin(psi)
    rect = plt_patches.Rectangle((-dx, -dz), l, w, linewidth=2,
                        edgecolor=color,
                        alpha=alpha,
                        facecolor=facecolor)
    t = matplotlib.transforms.Affine2D().translate(x, z)
    t = t.rotate_around(x, z, theta)
    t_start = ax.transData
    t_end =  t + t_start
    rect.set_transform(t_end)
    ax.add_patch(rect)

    
def draw_velocity_3d(track, color, ax=None):
    mean = track.mean
    if ax is None:
        ax = plt.gca()
    x, z, vx, vz = mean[0], mean[2], mean[7], mean[8]
    arr = plt.arrow(x, z, vx, vz,
                        color=color,
                        head_width=0.5,
                        head_length=0.5)
    ax.add_patch(arr)
                    
def draw_ellipse3d(covariance, x, y, color, ax=None):
    if np.any(np.isnan(covariance)):
        return
    if ax is None:
        ax = plt.gca()
    lambda_, v = np.linalg.eig(np.reshape(covariance[[0, 0, 2, 2], [0, 2, 0, 2]], (2,2)))
    lambda_ = np.sqrt(lambda_)
    idx = np.argsort(lambda_)[::-1]
    lambda_ = lambda_[idx]
    v = v[:, idx]
    nsigma = np.sqrt(5.99)
    ell = plt_patches.Ellipse(xy=(x,y)
                              , width= lambda_[0]*2*nsigma
                              , height=lambda_[1]*2*nsigma
                              , angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0]))
                              , edgecolor=color
                              , facecolor='none'
                              )
    ax.add_patch(ell)

def draw_track3d(track, color, ax=None):
    mu = track.mean
    draw_box3d(mu, color, 1, ax=ax)
    if ax is None:
        ax = plt.gca()
    x, z = mu[0], mu[2]
    ax.text(x, z, s = track.track_id, color='white', verticalalignment='top',
        bbox={'color': color, 'pad': 0})

    draw_ellipse3d(track.covariance, x, z, color, ax)
    draw_velocity_3d(track, color, ax)

def draw_detection3d(det, color, ax=None):
    draw_box3d(det.box_3d, color, 0.5, color, ax=ax)