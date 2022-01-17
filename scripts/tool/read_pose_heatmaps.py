'''
  @ Date: 2021/11/25 15:51
  @ Author: Zhao YaChen
'''
import cv2
import numpy as np
import struct
import os
from scipy.ndimage.filters import maximum_filter
import math
import json
from tqdm import tqdm
import copy
import socket

NMS_Threshold = 0.1
InterMinAbove_Threshold = 6
Inter_Threashold = 0.1

def show_parts_result_by_image(float_path, nx, ny, image_path, OUT_PATH , which):
    f = open(float_path, 'rb')  # save parts heatmaps and paf with size=(368*368)
    pic = np.zeros((nx, ny))  # float type
    pic_unit8 = np.zeros((nx, ny), dtype=np.uint8)
    f.seek(16)
    for i in range(nx):
        for j in range(ny):
            data = f.read(4)
            elem = struct.unpack('f', data)[0]
            # print(struct.unpack('f', data))
            pic[i][j] = elem
    f.close()

    # convert to unit8
    for h in range(pic.shape[0]):
        for w in range(pic.shape[1]):
            if pic[h, w] <= 0:
                pic_unit8[h, w] = 0
            else:
                pic_unit8[h, w] = int(pic[h, w] * 255)


    image = cv2.imread(image_path)
    cv2.imshow('image_ori', image)
    image = cv2.resize(image, (ny, 368), interpolation=cv2.INTER_CUBIC)

    # heatmap0
    img = pic_unit8[0:368, 0:ny]

    # convert to three channels
    img_heatmap0 = np.expand_dims(img, axis=2)
    img_heatmap0 = np.concatenate((img_heatmap0, img_heatmap0, img_heatmap0), axis=-1)

    image_heatmap_mix = np.zeros((image.shape[1], image.shape[0]), dtype=np.uint8)

    for i in range(1, which):  # show all parts heatmaps
        print(i)
        a = 368 * i
        b = 368 * (i + 1)
        img = pic_unit8[a:b, 0:ny]

        # convert to three channels
        img_heatmap = np.expand_dims(img, axis=2)
        img_heatmap = np.concatenate((img_heatmap, img_heatmap, img_heatmap), axis=-1)

        if i == 1:
            image_heatmap_mix = np.maximum(img_heatmap0, img_heatmap)
        else:
            image_heatmap_mix = np.maximum(image_heatmap_mix, img_heatmap)
    # save_path = os.path.join(OUT_PATH, 'heatmap'+str(i)+'.jpg')
    # print(save_path)
    # cv2.imwrite(save_path, mix)
    # cv2.imshow('mix', image)

    # cv2.waitKey(2)
    save_path = os.path.join(OUT_PATH, 'lala_0_0_heatmap_368x496_' + str(which) + '.jpg')
    print(save_path)
    cv2.imwrite(save_path, image_heatmap_mix)

    image_mix = cv2.addWeighted(image, 0.5, image_heatmap_mix, 0.5, 0.0)
    save_path = os.path.join(OUT_PATH, 'lala_0_1_heatmap_368x496_' + str(which) + '.jpg')
    print(save_path)
    cv2.imwrite(save_path, image_mix)


def show_pafs_reasult_by_image(float_path, nx, ny, image_path, OUT_PATH, which):
    f = open(float_path, 'rb')  # save parts heatmaps and paf with size=(368*368)
    pic = np.zeros((nx, ny))  # float type
    pic_unit8 = np.zeros((nx, ny), dtype=np.uint8)
    f.seek(16)
    for i in range(nx):
        for j in range(ny):
            data = f.read(4)
            elem = struct.unpack('f', data)[0]
            # print(struct.unpack('f', data))
            pic[i][j] = elem
    f.close()

    # convert to unit8
    for h in range(pic.shape[0]):
        for w in range(pic.shape[1]):
            if pic[h, w] <= 0:
                pic_unit8[h, w] = 0
            else:
                pic_unit8[h, w] = int(pic[h, w] * 255)

    image = cv2.imread(image_path)
    cv2.imshow('image_ori', image)
    image = cv2.resize(image, (ny, 368), interpolation=cv2.INTER_CUBIC)

    # heatmap0
    img = pic_unit8[0:368, 0:ny]

    # convert to three channels
    img_heatmap0 = np.expand_dims(img, axis=2)
    img_heatmap0 = np.concatenate((img_heatmap0, img_heatmap0, img_heatmap0), axis=-1)

    image_heatmap_mix = np.zeros((image.shape[1], image.shape[0]), dtype=np.uint8)

    for i in range(1, 2*(which+1)):  # show PAFx
        print(i)
        a = 368 * i
        b = 368 * (i + 1)
        img = pic_unit8[a:b, 0:ny]

        # convert to three channels
        img_heatmap = np.expand_dims(img, axis=2)
        img_heatmap = np.concatenate((img_heatmap, img_heatmap, img_heatmap), axis=-1)

        if i == 1:
            image_heatmap_mix = np.maximum(img_heatmap0, img_heatmap)
        else:
            image_heatmap_mix = np.maximum(image_heatmap_mix, img_heatmap)
    # save_path = os.path.join(OUT_PATH, 'heatmap'+str(i)+'.jpg')
    # print(save_path)
    # cv2.imwrite(save_path, mix)
    # cv2.imshow('mix', image)

    # cv2.waitKey(2)
    save_path = os.path.join(OUT_PATH, 'lala_0_0_PAFx_368x496_' + str(which) + '.jpg')
    print(save_path)
    cv2.imwrite(save_path, image_heatmap_mix)

    image_mix = cv2.addWeighted(image, 0.5, image_heatmap_mix, 0.5, 0.0)
    save_path = os.path.join(OUT_PATH, 'lala_0_1_PAFy_368x496_' + str(which) + '.jpg')
    print(save_path)
    cv2.imwrite(save_path, image_mix)

def non_max_suppression(heatmap, window_size=3, threshold=NMS_Threshold):
    heatmap[heatmap < threshold] = 0 # set low values to 0
    part_candidates = heatmap*(heatmap == maximum_filter(heatmap, footprint=np.ones((window_size, window_size))))
    return part_candidates

ShelfPairs = [
    (1, 8), (9, 10), (10, 11), (8, 9), (8, 12), (12, 13), (13, 14), (1, 2), (2, 3), (3, 4), (2, 17), (1, 5), (5, 6), (6, 7),
    (5, 18), (1, 0), (0, 15), (0, 16), (15, 17), (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)
] # = 26
ShelfNetwork = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23), (24, 25),
    (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36, 37), (38, 39), (40, 41), (42, 43), (44, 45), (46, 47), (48, 49),
    (50, 51)
] # = 26

def get_score(x1, y1, x2, y2, pafMatX, pafMatY):
    num_inter = 10
    dx, dy = x2 - x1, y2 - y1
    normVec = math.sqrt(dx ** 2 + dy ** 2)

    if normVec < 1e-4:
        return 0.0, 0

    vx, vy = dx / normVec, dy / normVec
    xs = np.arange(x1, x2, dx / num_inter) if x1 != x2 else np.full((num_inter, ), x1)
    ys = np.arange(y1, y2, dy / num_inter) if y1 != y2 else np.full((num_inter, ), y1)
    xs = (xs + 0.5).astype(int)
    ys = (ys + 0.5).astype(int)

    # without vectorization
    pafXs = np.zeros(num_inter)
    pafYs = np.zeros(num_inter)
    for idx, (mx, my) in enumerate(zip(xs, ys)):
        pafXs[idx] = pafMatX[my][mx]
        pafYs[idx] = pafMatY[my][mx]

    # vectorization slow?
    # pafXs = pafMatX[ys, xs]
    # pafYs = pafMatY[ys, xs]

    local_scores = pafXs * vx + pafYs * vy
    thidxs = local_scores > Inter_Threashold

    return sum(local_scores * thidxs)/10, sum(thidxs),

def estimate_pose_pair(coords, partIdx1, partIdx2, pafMatX, pafMatY):
    # connection_temp = []  # all possible connections
    origin_connection = []
    peak_coord1, peak_coord2 = coords[partIdx1], coords[partIdx2]

    for idx1, (y1, x1) in enumerate(zip(peak_coord1[0], peak_coord1[1])):
        for idx2, (y2, x2) in enumerate(zip(peak_coord2[0], peak_coord2[1])):
            if idx1 == idx2:
                score, count = get_score(x1, y1, x2, y2, pafMatX, pafMatY)
            else:
                score = 0.0
            origin_connection.append(score)
            # connection_temp.append({
            #     'score': score,
            #     'coord_p1': (x1, y1),
            #     'coord_p2': (x2, y2),
            #     'idx': (idx1, idx2), # connection candidate identifier
            #     'partIdx': (partIdx1, partIdx2),
            #     'uPartIdx': ('{}-{}-{}'.format(x1, y1, partIdx1), '{}-{}-{}'.format(x2, y2, partIdx2))
            # })

    # connection = []
    # used_idx1, used_idx2 = [], []
    # # sort possible connections by score, from maximum to minimum
    # for conn_candidate in sorted(connection_temp, key=lambda x: x['score'], reverse=True):
    #     # check not connected
    #     if conn_candidate['idx'][0] in used_idx1 or conn_candidate['idx'][1] in used_idx2:
    #         continue
    #     connection.append(conn_candidate)
    #     used_idx1.append(conn_candidate['idx'][0])
    #     used_idx2.append(conn_candidate['idx'][1])

    return origin_connection

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_coords(jsonname):
    """
    从openpose_part_candidates中json文件中读取”part_candidates,key_scale=3,range=[0,1]“
    """
    coords = []  # for each part index, it stores coordinates of candidates
    assert os.path.exists(jsonname), jsonname
    data = read_json(jsonname)
    origin_coords = data['part_candidates'][0]
    return origin_coords

def load_coords_net_resolution(jsonname):
    """
    从openpose_part_candidates中json文件中读取”part_candidates,key_scale=1“
    """
    coords = []  # for each part index, it stores coordinates of candidates
    assert os.path.exists(jsonname), jsonname
    data = read_json(jsonname)
    origin_coords = data['part_candidates'][0]
    for key, value in origin_coords.items():  # sum of joints
        '''
        key: str
        all_candidates_value: list{3n}
        '''
        # 是按照x1,y1,1_conf,x2,y2,2_conf来排列的
        # 要转化为(y1,y2)(x2,x1)
        all_candidates_value = np.array(value).reshape(-1, 3)
        y_seq = []
        x_seq = []
        for i, (x, y, conf) in enumerate(all_candidates_value):
            y_seq.append(y.astype(int))
            x_seq.append(x.astype(int))
        y_x_tuple = (y_seq, x_seq)
        coords.append(y_x_tuple)
    return origin_coords, coords

def load_coords_net_resolution_by_person_old(jsonname):
    """
   从openpose_part_candidates中json文件中按照person读取”part_candidates,key_scale=1“
   """
    origin_coords = []
    coords = []  # for each part index, it stores coordinates of candidates
    assert os.path.exists(jsonname), jsonname
    data = read_json(jsonname)

    temp_coords = []
    origin_people_coords = data['people']
    for people in origin_people_coords:
        all_candidates_value = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        temp_coords.append(copy.deepcopy(all_candidates_value))
        for i, value in enumerate(all_candidates_value):  # sum of joints
            # 组织成origin_coords格式
            # 是按照x,y,conf来排列的 是net_resolution大小
            # 要转化为0-1之间
            # 312.241, 77.2357, 0.853759
            value[0] = value[0]/net_resolution_X
            value[1] = value[1]/net_resolution_Y
            if len(origin_coords) > i:
                origin_coords[i] = np.append(origin_coords[i], value)
            else:
                origin_coords.append(value)

    # 组织成coords格式
    # 是按照x1,y1,1_conf,x2,y2,2_conf来排列的
    # 要转化为(y1,y2)(x2,x1)
    for i in range(len(temp_coords[0])):
        y_seq = []
        x_seq = []
        all_candidates_value = [row[i] for row in temp_coords]
        for i, (x, y, conf) in enumerate(all_candidates_value):
            y_seq.append(y.astype(int))
            x_seq.append(x.astype(int))
        y_x_tuple = (y_seq, x_seq)
        coords.append(y_x_tuple)
    return origin_coords, coords

def load_coords_net_resolution_by_person(origin_people_coords, net_resolution_X, net_resolution_Y):
    """
   origin_people_coords:(personNum, 25, 3)
   """
    origin_coords = []
    coords = []  # for each part index, it stores coordinates of candidates
    temp_coords = []
    for all_candidates_value in origin_people_coords:  # 25, 3
        temp_coords.append(copy.deepcopy(all_candidates_value))
        for i, value in enumerate(all_candidates_value):  # sum of joints
            # 组织成origin_coords格式
            # 是按照x,y,conf来排列的 是net_resolution大小
            # 要转化为0-1之间
            # 312.241, 77.2357, 0.853759
            value[0] = value[0]/net_resolution_X
            value[1] = value[1]/net_resolution_Y
            if len(origin_coords) > i:
                origin_coords[i] = np.append(origin_coords[i], value)
            else:
                origin_coords.append(value)

    # 组织成coords格式
    # 是按照x1,y1,1_conf,x2,y2,2_conf来排列的
    # 要转化为(y1,y2)(x2,x1)
    for i in range(len(temp_coords[0])):
        y_seq = []
        x_seq = []
        all_candidates_value = [row[i] for row in temp_coords]
        for i, (x, y, conf) in enumerate(all_candidates_value):
            y_seq.append(y.astype(int))
            x_seq.append(x.astype(int))
        y_x_tuple = (y_seq, x_seq)
        coords.append(y_x_tuple)
    return origin_coords, coords

def save_txt_for_detection(origin_coords, origin_connection_all, f, write_type):
    """
    保存为4d-association可以接受的/detxtion/0.txt的格式
    场景最大人数 所有帧数

    关节0的个数
    关节0的点
    ...

    根据PAFs计算出来的分数
    """
    '''
    coords:(25,tuple<array(y坐标),array(x坐标)>)
    origin_coords:dict{25}  <'0', list{part-candidates个数*3}>
    origin_connection_all:(26,1/2/4/6/9)
    '''
    coords_len = []
    for i, row in enumerate(origin_connection_all):
        for j, value in enumerate(row):
            if value < 0.1:
                origin_connection_all[i][j] = 0
    # 写入候选关节信息
    if write_type == 0:
        for key, value in origin_coords.items():  # sum of joints
            '''
            key: str
            all_candidates_value: list{3n}
            '''
            # 是按照x1,y1,1_conf,x2,y2,2_conf来排列的
            # 如果要转化为network大小的(y1,y2)(x2,x1)
            all_candidates_value = np.array(value).reshape(-1, 3)
            all_candidates_value_T = all_candidates_value.T
            coords_len.append(len(all_candidates_value))
            f.write(str(len(all_candidates_value))+'\n')
            for row in all_candidates_value_T:
                f.writelines('\t'.join(str(i) for i in row)+'\n')
    else:
        for key, value in enumerate(origin_coords):  # sum of joints
            '''
            key: str
            all_candidates_value: list{3n}
            '''
            # 是按照x1,y1,1_conf,x2,y2,2_conf来排列的
            # 如果要转化为network大小的(y1,y2)(x2,x1)
            all_candidates_value = np.array(value).reshape(-1, 3)
            all_candidates_value_T = all_candidates_value.T
            coords_len.append(len(all_candidates_value))
            f.write(str(len(all_candidates_value))+'\n')
            for row in all_candidates_value_T:
                f.writelines('\t'.join(str(i) for i in row)+'\n')


    # 写入根据PAFs计算出来的分数
    for (idx1, idx2), (i, connect) in zip(ShelfPairs, enumerate(origin_connection_all)):
        candidate_joint_1 = coords_len[idx1]
        candidate_joint_2 = coords_len[idx2]
        scores = np.array(connect).reshape(candidate_joint_1, candidate_joint_2)
        for row in scores:
            f.writelines('\t'.join(str(i) for i in row) + '\n')

# def save_txt_for_detection(origin_coords, origin_connection_all, s, write_type):
#     coords_len = []
#     for i, row in enumerate(origin_connection_all):
#         for j, value in enumerate(row):
#             if value < 0.1:
#                 origin_connection_all[i][j] = 0
#     # 写入候选关节信息
#     if write_type == 0:
#         for key, value in origin_coords.items():  # sum of joints
#             '''
#             key: str
#             all_candidates_value: list{3n}
#             '''
#             # 是按照x1,y1,1_conf,x2,y2,2_conf来排列的
#             # 如果要转化为network大小的(y1,y2)(x2,x1)
#             all_candidates_value = np.array(value).reshape(-1, 3)
#             all_candidates_value_T = all_candidates_value.T
#             coords_len.append(len(all_candidates_value))
#             s += str(len(all_candidates_value))+'\n'
#             for row in all_candidates_value_T:
#                 s += '\t'.join(str(i) for i in row)+'\n'
#     else:
#         for key, value in enumerate(origin_coords):  # sum of joints
#             '''
#             key: str
#             all_candidates_value: list{3n}
#             '''
#             # 是按照x1,y1,1_conf,x2,y2,2_conf来排列的
#             # 如果要转化为network大小的(y1,y2)(x2,x1)
#             all_candidates_value = np.array(value).reshape(-1, 3)
#             all_candidates_value_T = all_candidates_value.T
#             coords_len.append(len(all_candidates_value))
#             s += str(len(all_candidates_value))+'\n'
#             for row in all_candidates_value_T:
#                 s += '\t'.join(str(i) for i in row)+'\n'
#
#
#     # 写入根据PAFs计算出来的分数
#     for (idx1, idx2), (i, connect) in zip(ShelfPairs, enumerate(origin_connection_all)):
#         candidate_joint_1 = coords_len[idx1]
#         candidate_joint_2 = coords_len[idx2]
#         scores = np.array(connect).reshape(candidate_joint_1, candidate_joint_2)
#         for row in scores:
#             s += '\t'.join(str(i) for i in row) + '\n'
#
#     return s


def estimate_pose(heatMat, pafMat):
    """
    heatMat:(height, width, n_parts)
    pafMat:(height, width, n_pairs*2)

    heatMat:(n_parts, height, width)
    pafMat:(n_pairs*2, height, width)
    """
    # reliability issue.
    heatMat = heatMat - heatMat.min(axis=1).min(axis=1).reshape(25, 1, 1)
    heatMat = heatMat - heatMat.min(axis=2).reshape(25, heatMat.shape[1], 1)

    _NMS_Threshold = max(np.average(heatMat) * 4.0, NMS_Threshold)
    _NMS_Threshold = min(_NMS_Threshold, 0.3)

    coords = []  # for each part index, it stores coordinates of candidates
    for heatmap in heatMat:
        part_candidates = non_max_suppression(heatmap, 5, _NMS_Threshold)
        coords.append(np.where(part_candidates >= _NMS_Threshold))

    connection_all = []  # all connections detected. no information about what humans they belong to
    origin_connection_all = []
    for (idx1, idx2), (paf_x_idx, paf_y_idx) in zip(ShelfPairs, ShelfNetwork):
        connection, origin_connection = estimate_pose_pair(coords, idx1, idx2, pafMat[paf_x_idx], pafMat[paf_y_idx])
        connection_all.extend(connection)
        origin_connection_all.append(origin_connection)

    return origin_connection_all, connection_all

def estimate_pose_by_part_condidates(partCandidatesName, pafMat):
    origin_coords, coords = load_coords_net_resolution(partCandidatesName)
    connection_all = []  # all connections detected. no information about what humans they belong to
    origin_connection_all = []
    for (idx1, idx2), (paf_x_idx, paf_y_idx) in zip(ShelfPairs, ShelfNetwork):
        connection, origin_connection = estimate_pose_pair(coords, idx1, idx2, pafMat[paf_x_idx], pafMat[paf_y_idx])
        connection_all.extend(connection)
        origin_connection_all.append(origin_connection)

    return origin_connection_all, connection_all

def estimate_pose_by_given_part(origin_coords, coords, pafMat):
    # connection_all = []  # all connections detected. no information about what humans they belong to
    origin_connection_all = []
    for (idx1, idx2), (paf_x_idx, paf_y_idx) in zip(ShelfPairs, ShelfNetwork):
        origin_connection = estimate_pose_pair(coords, idx1, idx2, pafMat[paf_x_idx], pafMat[paf_y_idx])
        # connection_all.extend(connection)
        origin_connection_all.append(origin_connection)

    return origin_connection_all, origin_coords


if __name__ == '__main__':
    # which = 17
    # show_parts_result_by_image(heatMapFullPath, which*net_resolution_Y, net_resolution_X, image_path, OUT_PATH, which)
    # show_pafs_reasult_by_image(heatMapFullPath, (which+1)*2*net_resolution_Y, net_resolution_X, image_path, OUT_PATH, which)
    # partCandidatesPath = "D:/Desktop/4d_association/4d_association_data/" + dataset + "/openpose_part_candidates/{}".format(cam)
    # partCandidatesNetResolutionPath = "D:/Desktop/4d_association/4d_association_data/" + dataset + "/openpose_part_candidates/{}_net_resolution".format(cam)
    # heatMapOnlyAddPAFsFullPath = "D:/Desktop/4d_association/4d_association_data/shelf/output_heatmaps_folder/0_only_add_PAFs_float/000000_pose_heatmaps.float"
    # heatMapOnlyAddPartsFullPath = "D:/Desktop/4d_association/4d_association_data/shelf/output_heatmaps_folder/0_only_add_parts_float/000000_pose_heatmaps.float"
    # image_path = "D:/Desktop/4d_association/4d_association_data/" + dataset + "/images/0/000500.jpg"
    # OUT_PATH = "D:/Desktop/4d_association/4d_association_data/" + dataset + "/out_images/0"

    # origin_connection_all, connection_all = estimate_pose(heatMat, pafMat)
    # origin_coords = load_coords(partCandidatesName)
    # origin_connection_all, connection_all = estimate_pose_by_part_condidates(partCandidatesNetResulutionName, pafMat)
    # save_txt_for_detection(origin_coords, origin_connection_all, f, 0)


    # dataset = 'shelf'
    # socket.AF_INET用于服务器与服务器之间的网络通信
    # socket.SOCK_STREAM代表基于TCP的流式socket通信
    # 1. 加载套接字库，创建套接字(WSAStartup()/socket())；
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接服务端
    # 2. 向服务器发出连接请求(connect())；
    address_server = ('127.0.0.1', 8010)
    sock.connect(address_server)
    # 开始准备每一帧的数据
    dataset = 'markless_multiview_data/seq_1'
    OUT_TXT_PATH = "D:/Desktop/4d_association/4d_association_data/" + dataset + "/detections"
    start = 500
    frames = 300
    for nf in tqdm(range(start, start+frames)):
        outTxtName = os.path.join(OUT_TXT_PATH, '{}.txt'.format(nf-start))
        for cam in range(0, 6):
            partPath = "D:/Desktop/4d_association/4d_association_data/" + dataset + "/openpose_part_candidates/{}_net_resolution".format(cam)
            partName = os.path.join(partPath, '{:06d}_keypoints.json'.format(nf))
            heatMapFullPath = "D:/Desktop/4d_association/4d_association_data/" + dataset + "/output_heatmaps_folder/{}_float_net_resolution".format(cam)
            heatMapFullName = os.path.join(heatMapFullPath, '{:06d}_pose_heatmaps.float'.format(nf))

            x = np.fromfile(heatMapFullName, dtype=np.float32)
            x_new = np.reshape(x[4:], (77, net_resolution_Y, net_resolution_X))
            # heatMat = x_new[0:25, :, :]
            pafMat = x_new[25:, :, :]
            '''
            coords:(25,tuple<array(y坐标),array(x坐标)>)
            origin_coords:(25,3*3)
            origin_connection_all:(26,3*3PAF分数矩阵)
            '''
            origin_coords, coords = load_coords_net_resolution_by_person_old(partName)
            # origin_coords, coords = load_coords_net_resolution_by_person(keypoint)
            origin_connection_all, origin_coords = estimate_pose_by_given_part(origin_coords, coords, pafMat)
            with open(outTxtName, 'a') as f:  # 设置文件对象
                # f.write('4\t300\n')  # 将字符串写入文件中
                save_txt_for_detection(origin_coords, origin_connection_all, f, 1)
        f.close()
        # print('第{}帧FINISH!'.format(nf))
        # 3. 和服务器端进行通信(send()/recv())；
        # fp = open(outTxtName, 'rb')
        # data = fp.read()
        # 如果只发送txt文件名称
        data = str.encode(str(nf-start)+'\n')
        sock.send(data)
        # print('{0} fileName send over...'.format(data))
    # 结束标志：发送一个Q
    sock.send(str.encode('Q'))
    # 4. 关闭套接字，关闭加载的套接字库(closesocket()/WSACleanup())
    sock.close()
    cv2.destroyAllWindows()
    # # Load custom float format - Example in Python, assuming a (18 x 300 x 500) size Array
    # x = np.fromfile(heatMapFullPath, dtype=np.float32)
    # x_PAFs = np.fromfile(heatMapOnlyAddPAFsFullPath, dtype=np.float32)
    # x_parts = np.fromfile(heatMapOnlyAddPartsFullPath, dtype=np.float32)
    # assert x[0] == 3 # First parameter saves the number of dimensions (18x300x500 = 3 dimensions)
    # shape_x = x[1:1+int(x[0])]
    # assert len(shape_x[0]) == 3 # Number of dimensions
    # assert shape_x[0] == 18 # Size of the first dimension
    # assert shape_x[1] == 300 # Size of the second dimension
    # assert shape_x[2] == 500 # Size of the third dimension
    # arrayData = x[1+int(round(x[0])):]
    # print("lalal")

