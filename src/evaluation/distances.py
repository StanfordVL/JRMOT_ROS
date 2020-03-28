"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import numpy as np
import pdb

def norm2squared_matrix(objs, hyps, max_d2=float('inf')):
    """Computes the squared Euclidean distance matrix between object and hypothesis points.

    Params
    ------
    objs : NxM array
        Object points of dim M in rows
    hyps : KxM array
        Hypothesis points of dim M in rows

    Kwargs
    ------
    max_d2 : float
        Maximum tolerable squared Euclidean distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to +inf

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0,0))

    assert hyps.shape[1] == objs.shape[1], "Dimension mismatch"

    C = np.empty((objs.shape[0], hyps.shape[0]))

    for o in range(objs.shape[0]):
        for h in range(hyps.shape[0]):
            e = objs[o] - hyps[h]
            C[o, h] = e.dot(e)

    C[C > max_d2] = np.nan
    return C


def iou_matrix(objs, hyps, max_iou=1.):
    """Computes 'intersection over union (IoU)' distance matrix between object and hypothesis rectangles.

    The IoU is computed as

        IoU(a,b) = 1. - isect(a, b) / union(a, b)

    where isect(a,b) is the area of intersection of two rectangles and union(a, b) the area of union. The
    IoU is bounded between zero and one. 0 when the rectangles overlap perfectly and 1 when the overlap is
    zero.

    Params
    ------
    objs : Nx4 array
        Object rectangles (x,y,w,h) in rows
    hyps : Kx4 array
        Hypothesis rectangles (x,y,w,h) in rows

    Kwargs
    ------
    max_iou : float
        Maximum tolerable overlap distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to 0.5

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0,0))

    assert objs.shape[1] == 4
    assert hyps.shape[1] == 4

    br_objs = objs[:, :2] + objs[:, 2:]
    br_hyps = hyps[:, :2] + hyps[:, 2:]

    C = np.empty((objs.shape[0], hyps.shape[0]))

    for o in range(objs.shape[0]):
        for h in range(hyps.shape[0]):
            isect_xy = np.maximum(objs[o, :2], hyps[h, :2])
            isect_wh = np.maximum(np.minimum(br_objs[o], br_hyps[h]) - isect_xy, 0)
            isect_a = isect_wh[0]*isect_wh[1]
            union_a = objs[o, 2]*objs[o, 3] + hyps[h, 2]*hyps[h, 3] - isect_a
            if union_a != 0:
                C[o, h] = 1. - isect_a / union_a
            else:
                C[o, h] = np.nan

    C[C > max_iou] = np.nan
    return C


def find_area(vertices):
    area = 0
    for i in range(len(vertices)):
        area += vertices[i][0]*(vertices[(i+1)%len(vertices)][1] - vertices[i-1][1])
    return 0.5*abs(area)

def get_angle(p):
    x, y = p
    angle = np.arctan2(y,x)
    if angle < 0:
        angle += np.pi*2
    return angle

def clip_polygon(box1, box2):
    #clips box 1 by the edges in box2
    x,y,z,l,h,w,theta = box2
    theta = -theta

    box2_edges = np.asarray([(-np.cos(theta), -np.sin(theta), l/2-x*np.cos(theta)-z*np.sin(theta)),
                    (-np.sin(theta), np.cos(theta), w/2-x*np.sin(theta)+z*np.cos(theta)),
                    (np.cos(theta), np.sin(theta), l/2+x*np.cos(theta)+z*np.sin(theta)),
                    (np.sin(theta), -np.cos(theta), w/2+x*np.sin(theta)-z*np.cos(theta))])
    x,y,z,l,h,w,theta = box1
    theta = -theta

    box1_vertices = [(x+l/2*np.cos(theta)-w/2*np.sin(theta), z+l/2*np.sin(theta)+w/2*np.cos(theta)),
                        (x+l/2*np.cos(theta)+w/2*np.sin(theta), z+l/2*np.sin(theta)-w/2*np.cos(theta)),
                        (x-l/2*np.cos(theta)-w/2*np.sin(theta), z-l/2*np.sin(theta)+w/2*np.cos(theta)),
                        (x-l/2*np.cos(theta)+w/2*np.sin(theta), z-l/2*np.sin(theta)-w/2*np.cos(theta))]
    out_vertices = sort_points(box1_vertices, (x, z))
    for edge in box2_edges:
        vertex_list = out_vertices.copy()
        out_vertices = []
        for idx, current_vertex in enumerate(vertex_list):
            previous_vertex = vertex_list[idx-1]
            if point_inside_edge(current_vertex, edge):
                if not point_inside_edge(previous_vertex, edge):
                    out_vertices.append(compute_intersection_point(previous_vertex, current_vertex, edge))
                out_vertices.append(current_vertex)
            elif point_inside_edge(previous_vertex, edge):
                out_vertices.append(compute_intersection_point(previous_vertex, current_vertex, edge))
    to_remove = []
    for i in range(len(out_vertices)):
        if i in to_remove:
            continue
        for j in range(i+1, len(out_vertices)):
            if abs(out_vertices[i][0] - out_vertices[j][0]) < 1e-6 and abs(out_vertices[i][1] - out_vertices[j][1]) < 1e-6:
                to_remove.append(j)
    out_vertices = sorted([(v[0]-x, v[1]-z) for i,v in enumerate(out_vertices) if i not in to_remove], key = lambda p: get_angle((p[0],p[1])))
    return out_vertices

def sort_points(pts, center):
    x, z = center
    sorted_pts = sorted([(i, (v[0]-x, v[1]-z)) for i,v in enumerate(pts)], key = lambda p: get_angle((p[1][0],p[1][1])))
    idx, _ = zip(*sorted_pts)
    return [pts[i] for i in idx]

def compute_intersection_point(pt1, pt2, line1):
    if pt1[0] == pt2[0]:
        slope = np.inf
    else:
        slope = (pt1[1]-pt2[1])/(pt1[0] - pt2[0])
    if np.isinf(slope):
        line2 = (1, 0, pt1[0])
    else:
        line2 = (slope, -1, pt1[0]*slope-pt1[1])
    # print("Line1:", line1)
    # print("Line2:", line2)
    if line1[1] == 0:
        x = line1[2]/line1[0]
        y = (line2[2] - line2[0]*x)/line2[1]
    elif line1[0] == 0:
        y = line1[2]/line1[1]
        x = (line2[2] - line2[1]*y)/line2[0]
    elif line2[1] == 0:
        x = pt1[0]
        y = (line1[2]-x*line1[0])/line1[1]
    else:
        tmp_line = (line2 - line1*(line2[1]/line1[1]))
        x = tmp_line[2]/tmp_line[0]
        y = (line2[2] - line2[0]*x)/line2[1]
    return (x,y)

def point_inside_edge(pt, edge):
    lhs = pt[0]*edge[0] + pt[1]*edge[1]
    if lhs < edge[2] - 1e-6:
        return True
    else:
        return False


def iou_matrix_3d(objs, hyps, max_iou=1.):
    """Computes 'intersection over union (IoU)' distance matrix between object and hypothesis rectangles.

    The IoU is computed as

        IoU(a,b) = 1. - isect(a, b) / union(a, b)

    where isect(a,b) is the area of intersection of two rectangles and union(a, b) the area of union. The
    IoU is bounded between zero and one. 0 when the rectangles overlap perfectly and 1 when the overlap is
    zero.

    Params
    ------
    objs : Nx4 array
        Object rectangles (x,y,w,h) in rows
    hyps : Kx4 array
        Hypothesis rectangles (x,y,w,h) in rows

    Kwargs
    ------
    max_iou : float
        Maximum tolerable overlap distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to 0.5

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    objs = np.atleast_2d(objs).astype(float)
    hyps = np.atleast_2d(hyps).astype(float)

    if objs.size == 0 or hyps.size == 0:
        return np.empty((0,0))
    assert objs.shape[1] == 7
    assert hyps.shape[1] == 7

    C = np.empty((objs.shape[0], hyps.shape[0]))
    for o in range(objs.shape[0]):
        for h in range(hyps.shape[0]):
            base_area = find_area(clip_polygon(objs[o], hyps[h]))
            height = max(objs[o][1], hyps[h][1]) - min(objs[o][1] - objs[o][4], hyps[h][1]-hyps[h][4])
            intersect = base_area*height
            union = objs[o][3]*objs[o][4]*objs[o][5] + hyps[h][3]*hyps[h][4]*hyps[h][5] - intersect
            if union != 0:
                C[o, h] = 1. - intersect / union
            else:
                C[o, h] = np.nan
    C[C > max_iou] = np.nan
    return C
