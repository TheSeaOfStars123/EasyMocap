'''
  @ Date: 2021/12/20 19:28
  @ Author: Zhao YaChen
'''
import numpy as np

def calc_ray(uv_homo, cam):
    eiRtKi = cam['eiRtKi']
    ray = -eiRtKi @ uv_homo
    return ray / np.linalg.norm(ray)

def calc_ray_points(points, cam):
    conf = points[..., 2:]
    points_homo = np.hstack([points[..., :2], np.ones([points.shape[0], 1])])
    points_homo2 = np.concatenate([points[..., :2], np.ones_like(conf)], axis=-1)
    res = []
    for point in points_homo:
        res.append(calc_ray(point, cam).transpose())
    res = np.hstack([np.array(res), conf])
    return res


def point2line_dist(pA, pB, ray):
    return np.linalg.norm(np.cross((pA - pB), ray))


def line2line_dist(camA, raysA, camB, raysB):
    camAPos = camA['eiPos']
    camBPos = camB['eiPos']
    conf = np.sqrt(raysA[..., -1] * raysB[..., -1])    #(4, 3, 15)
    raysA = raysA[:, 0]
    raysB = raysB[0, :]
    dist = np.zeros((raysA.shape[0], raysB.shape[0]))
    for view0, p0 in enumerate(raysA):  # (4, 15, 4)
        for view1, p1 in enumerate(raysB):  #(3, 15, 4)
            for joint in range(p0.shape[0]):  # 15个关节
                score_sum = 0
                rayA = p0[joint][:3]
                rayB = p1[joint][:3]
                if np.abs(np.dot(rayA, rayB)) < 1e-05:
                    score = point2line_dist(camAPos, camBPos, rayA)
                else:
                    cross = np.cross(rayA, rayB)
                    cross = cross / np.linalg.norm(cross)
                    score = np.abs(np.dot((camAPos-camBPos).T, cross))
                score_sum += score
            dist[view0][view1] = score_sum
    dist = np.sum(dist * conf, axis=-1) / (1e-5 + conf.sum(axis=-1))  # (4, 3)
    dist[conf.sum(axis=-1) < 0.1] = 1e5
    return dist






