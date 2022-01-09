# @Time : 2021/11/14 9:11 PM
# @Author : zyc
# @File : mvmp_greedy.py
# @Title :
# @Description :
from easymocap.dataset import CONFIG
from easymocap.affinity.affinity import ComposedAffinity
from easymocap.affinity import plucker
from easymocap.affinity.plucker import computeRay, dist_pl_pointwise, dist_pl_pointwise_conf, convert_Qc2Qg
from easymocap.affinity.jointsRay import calc_ray_points
from easymocap.assignment.associate import simple_associate
from easymocap.assignment.group import PeopleGroup
from easymocap.mytools import Timer, read_annot
from easymocap.mytools.reconstruction import batch_triangulate, projectN3

from tqdm import tqdm
import numpy as np
import os
import cv2


def compute_correspondence_3D_3D(k3d_i, k3d_j):
    max_dist = 0.15
    max_dist_step = 0.01
    """
    计算3D点之间的距离
    """
    conf = np.sqrt(k3d_i[:, 3] * k3d_j[:, 3])
    d_ij = np.linalg.norm(k3d_i[:, :3] - k3d_j[:, :3], axis=1)
    a_ij = 1 - d_ij / (max_dist + max_dist_step)
    a_ij[a_ij < 0] = 0
    weight = (conf * a_ij).sum(axis=0) / (1e-4 + conf.sum(axis=0))
    return weight

def compute_correspondence_2D_3D(keypoint2d_line, keypoint3d, R, T):
    """
    计算3D点到2D极线的点到线上的距离
    keypoint2d_line: {1, 15, 7}
    """
    # keypoint3d[:15, :3] = convert_Qc2Qg(keypoint3d[:15, :3], R, T)
    dist = dist_pl_pointwise_conf(keypoint3d[:15, :], keypoint2d_line)
    return dist

def set_keypoints2d(indices, annots, Pall, dimGroups):
    Vused = np.where(indices != -1)[0]
    if len(Vused) < 1:
        return [], [], [], []
    keypoints2d = np.stack([annots[nv][indices[nv] - dimGroups[nv]]['keypoints'].copy() for nv in Vused])
    bboxes = np.stack([annots[nv][indices[nv] - dimGroups[nv]]['bbox'].copy() for nv in Vused])
    Pused = Pall[Vused]
    return keypoints2d, bboxes, Pused, Vused


def greedy_graph(result, affinity, dimGroups, nf, Pall, lPluckers, M_constr=None,):
    maxid = 0
    frames_of_id = {}
    log = print
    log = lambda x: x
    pemi = np.zeros((affinity.shape[0], affinity.shape[1]))
    views = np.zeros(dimGroups[-1], dtype=np.int)
    for nv in range(len(dimGroups) - 1):  # dimGroups:[0,4,7,11...]
        views[dimGroups[nv]:dimGroups[nv + 1]] = nv  # views:[0,0,0,0,1,1,1,2,2,2,2...]

    # 将affinity对角线设置为1
    N = affinity.shape[0]  # 14
    index_diag = np.arange(N)  # [0 1 2 3 ... 13]
    # X[index_diag, index_diag] = 0.
    if M_constr is None:
        M_constr = np.ones_like(affinity)
        for i in range(len(dimGroups) - 1):
            M_constr[dimGroups[i]:dimGroups[i+1], dimGroups[i]:dimGroups[i+1]] = 0
        M_constr[index_diag, index_diag] = 1
    affinity[index_diag, index_diag] = 1.
    pemi[index_diag, index_diag] = 1.
    affinity = affinity * M_constr

    # 第一步：按照分数从大到小排序
    # 第二步：如果候选edge对着的两条边的指向都没有id,那么就建立连接
    # 第三步：已知1和5建立连接
    #   想建立5和10的连接
    #       如果10没有连接 那么只需要考虑1和10之间的分数是不是有效连接，如果不是就不能建立连接。如果是可以建立连接
    #       如果10有连接 并且对应的是另一个人 那么5和10就不能建立连接。如果对应的是同一个人 可以建立连接
    is_valid_affinity = affinity.copy()
    is_valid_affinity[pemi == 1] = 0
    # 第一次迭代计算的是2D-2D的Affinity
    # 循环退出的条件是没有更多有效的连接
    has_new_edge = False
    while np.max(is_valid_affinity) > 0.1:
        # 当前分数最高的是候选边
        edge_candidates = np.where(is_valid_affinity == np.max(is_valid_affinity))
        edge_candidate_x = edge_candidates[0][0]
        edge_candidate_y = edge_candidates[1][0]
        nf0 = views[edge_candidate_x]  # 边的左边位于哪个视图
        ni0 = edge_candidate_x - dimGroups[nf0]  # 边的左边位于这个视图的哪个人
        nf1 = views[edge_candidate_y]  # 边的右边位于哪个视图
        ni1 = edge_candidate_y - dimGroups[nf1]  # 边的右边位于这个视图的哪个人
        # 先看有没有连接
        id0 = result[nf0][ni0]['id']
        id1 = result[nf1][ni1]['id']
        # directly assign
        if id0 == -1 and id1 == -1:
            result[nf0][ni0]['id'] = maxid
            result[nf1][ni1]['id'] = maxid
            log('Create person {}'.format(maxid))
            frames_of_id[maxid] = {nf0: ni0, nf1: ni1}
            maxid += 1
            pemi[edge_candidate_x][edge_candidate_y] = 1
            pemi[edge_candidate_y][edge_candidate_x] = 1
            has_new_edge = True
        elif id0 != -1 and id1 == -1:
            #  5---10   27 10和27想要连接，判断27和(10之前建立连接)5、8、17之间是否有效(分数大于0.5)
            #  8---10
            # 17---10
            can_connect = True
            # 遍历10之前的连接
            before_edges = np.where(pemi[edge_candidate_x] == 1)[0]
            for i, edge in enumerate(before_edges):
                # 判断分数是否有效 如果无效 先跳过这次连接
                if affinity[edge][edge_candidate_y] < 0.1:
                    can_connect = False
                    break
            if can_connect:
                # 还要判断id0中是否已经与当前视图nf1中其他人次对应：
                if nf1 in frames_of_id[id0].keys():
                    log('Merge conflict')
                    # 出现冲突之后就把原来id0中对应视图nf1中其他人次删除
                    # log('Merge person {}'.format(maxid))
                    # 去掉原来的连接
                    pemi[dimGroups[nf1] + frames_of_id[id0][nf1]][result[nf1][frames_of_id[id0][nf1]]] = 0
                    pemi[result[nf1][frames_of_id[id0][nf1]]][dimGroups[nf1] + frames_of_id[id0][nf1]] = 0
                # 建立新的连接
                result[nf1][ni1]['id'] = id0
                frames_of_id[id0][nf1] = ni1
                pemi[edge_candidate_x][edge_candidate_y] = 1
                pemi[edge_candidate_y][edge_candidate_x] = 1
                has_new_edge = True
            else:
                # 此时不连接 寻找下一个候选连接
                is_valid_affinity[edge_candidate_x][edge_candidate_y] = 0
                is_valid_affinity[edge_candidate_y][edge_candidate_x] = 0
        elif id0 == -1 and id1 != -1:
            #     5---8
            # 0   5---10 0和5想要连接，判断0和(5之前建立连接)8、10、17之间是否有效(分数大于0.5)
            #     5---17
            # 还要判断id1中是否已经与当前视图nf0中其他人次对应：
            if nf0 in frames_of_id[id1].keys():
                log('Merge conflict')
            result[nf0][ni0]['id'] = id1
            frames_of_id[id1][nf0] = ni0
            pemi[edge_candidate_x][edge_candidate_y] = 1
            pemi[edge_candidate_y][edge_candidate_x] = 1
        elif id0 == id1 and id0 != -1:
            pemi[edge_candidate_x][edge_candidate_y] = 1
            pemi[edge_candidate_y][edge_candidate_x] = 1
            has_new_edge = True
        # merge
        elif id0 != id1:
            common = frames_of_id[id0].keys() & frames_of_id[id1].keys()
            for key in common:  # conflict
                if frames_of_id[id0][key] == frames_of_id[id1][key]:
                    pass
                else:
                    is_valid_affinity[edge_candidate_x][edge_candidate_y] = 0
                    is_valid_affinity[edge_candidate_y][edge_candidate_x] = 0
                    break
            else:  # merge
                log('Merge {} to {}'.format(id1, id0))
                for key in frames_of_id[id1].keys():
                    result[key][frames_of_id[id1][key]]['id'] = id0
                    frames_of_id[id0][key] = frames_of_id[id1][key]
                    pemi[edge_candidate_x][edge_candidate_y] = 1
                    pemi[edge_candidate_y][edge_candidate_x] = 1
                    has_new_edge = True
                frames_of_id.pop(id1)
                continue
            log('Conflict; not merged')
        # 如果有新的边连接 重新计算Affinity矩阵
        if has_new_edge:
            # 对新连接涉及到的这个人重新计算批量三角化
            # pemi[1][8] = 1
            # pemi[8][1] = 1
            # pemi[1][13] = 1
            # pemi[13][1] = 1
            Vused = np.where(pemi[edge_candidate_x] == 1)[0]
            if len(Vused) > 1:
                id = result[nf0][ni0]['id']
                keypoints2d = np.stack(
                    result[key][frames_of_id[id][key]]['keypoints'] for key in frames_of_id[id].keys())
                Pused = np.stack(Pall[key] for key in frames_of_id[id].keys())
                keypoints3d = batch_triangulate(keypoints2d, Pused)  # (25, 4)
                # 给当前人 对应视图的对应人次更新keypoints3d
                for key in frames_of_id[id].keys():
                    result[key][frames_of_id[id][key]]['keypoints3d'] = keypoints3d
                # 重新计算Affinity矩阵 计算那一列的数据
                new_affinity = affinity
                scores = []
                for row in range(dimGroups[-1]):
                    if row in Vused:
                        scores.append(1)
                    # elif
                    else:
                        # 将row转化成第几视图的第几人
                        nf_row = views[row]
                        ni_row = row - dimGroups[nf_row]
                        # 判断是改用3D-2D 还是 3D-3D的计算方式
                        if result[nf_row][ni_row]['id'] != -1:
                            weight = compute_correspondence_3D_3D(keypoints3d, result[nf_row][ni_row]['keypoints3d'])
                            scores.append(weight)
                            # scores.append(0)
                        else:
                            distance = compute_correspondence_2D_3D(lPluckers[nf_row][ni_row], keypoints3d,
                                                                    dataset.cameras[dataset.cams[nf_row]]['R'],
                                                                    dataset.cameras[dataset.cams[nf_row]]['T'])
                            if distance > 0.75:
                                distance = 0.75
                            score = 1 - distance / 0.75
                            scores.append(score)
                # 将计算出这一列的数据更新到其他列
                for v in Vused:
                    scores = np.array(scores)
                    new_affinity[:, v] = scores
                    new_affinity[v, :] = scores.T
                affinity = new_affinity
                affinity[index_diag, index_diag] = 1.
                affinity = affinity * M_constr
                is_valid_affinity = affinity.copy()
            # 将已经判断过的边不再考虑
            is_valid_affinity[pemi == 1] = 0
    return pemi, result


def mvposev1(dataset, args, cfg):
    dataset.no_img = not (args.vis_det or args.vis_match or args.vis_repro or args.ret_crop)
    start, end = args.start, min(args.end, len(dataset))
    affinity_model = ComposedAffinity(cameras=dataset.cameras, basenames=dataset.cams, cfg=cfg.affinity)
    group = PeopleGroup(Pall=dataset.Pall, cfg=cfg.group)
    if args.vis3d:
        from easymocap.socket.base_client import BaseSocketClient
        vis3d = BaseSocketClient(args.host, args.port)
    for nf in tqdm(range(start, end), desc='reconstruction'):
        group.clear()
        with Timer('load data', not args.time):
            # 取第nf帧的多个视图
            images, annots = dataset[nf]
            # 取第nf帧的多个视图对应的json文件
            result = []
            '''
            result:[第n-th视图
                [第i-th人次
                    {
                    'bbox':
                    'keypoints'：
                    'isKeyframe':
                    'id': 用id是否是-1来判断是否计算出3d坐标
                    'index':
                    'img':
                    'keypoints3d'：
                    }
                ]
            ]
            '''
            # 对8个相机依次加入
            for cam in dataset.cams:
                # 读img
                imgname = os.path.join(dataset.image_root, cam, dataset.imagelist[cam][nf])
                assert os.path.exists(imgname), imgname
                img = cv2.imread(imgname)
                # 读annots
                annname = os.path.join(dataset.annot_root, cam, dataset.annotlist[cam][nf])
                assert os.path.exists(annname), annname
                assert dataset.imagelist[cam][nf].split('.')[0] == dataset.annotlist[cam][nf].split('.')[0]
                infos = read_annot(annname, dataset.kpts_type)
                for n, info in enumerate(infos):
                    info['id'] = -1
                    info['index'] = n
                    info['img'] = img
                    info['keypoints3d'] = []
                result.append(infos)
            # 准备每个视图 每个人 每个关节的 2D plucker线
            lPluckers = []
            for nv, annot in enumerate(annots):
                cam = dataset.cameras[dataset.cams[nv]]
                pluckers = []
                for det in annot:
                    lines = computeRay(det['keypoints'][None, :15, :],
                                       cam['invK'], cam['R'], cam['T'])[0]
                    pluckers.append(lines)
                if len(pluckers) > 0:
                    pluckers = np.stack(pluckers)
                lPluckers.append(pluckers)
            # 准备每个视图 每个人 每个关节的 JointRay线
            m_jointRays = []
            for nv, annot in enumerate(annots):
                cam = dataset.cameras[dataset.cams[nv]]
                jointRays = []
                for det in annot:
                    rays = calc_ray_points(det['keypoints'][:15, :], cam)
                    jointRays.append(rays)
                if len(jointRays) > 0:
                    jointRays = np.stack(jointRays)
                m_jointRays.append(jointRays)
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        # 计算不同视角的检测结果的affinity
        with Timer('compute affinity', not args.time):
            affinity, dimGroups = affinity_model(annots, images=images)
            # 根据Real-time论文中的贪心算法计算置换矩阵
            pemi, result = greedy_graph(result, affinity, dimGroups, nf, dataset.Pall, lPluckers)
        with Timer('associate', not args.time):
            group = simple_associate(annots, pemi, dimGroups, dataset.Pall, group, cfg=cfg.associate)
            results = group
        if args.vis_match:
            dataset.vis_detections(images, annots, nf, mode='match', sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, results, nf, sub_vis=args.sub_vis)
        dataset.write_keypoints2d(annots, nf)
        dataset.write_keypoints3d(results, nf)
        if args.vis3d:
            vis3d.send(group.results)
    Timer.report()


if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser

    parser = load_parser()
    parser.add_argument('--vis_match', action='store_true')
    parser.add_argument('--time', action='store_true')
    parser.add_argument('--vis3d', action='store_true')
    parser.add_argument('--ret_crop', action='store_true')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument("--host", type=str, default='none')  # cn0314000675l
    parser.add_argument("--port", type=int, default=9999)
    args = parse_parser(parser)
    from easymocap.config.mvmp1f import Config

    cfg = Config.load(args.cfg, args.cfg_opts)
    # Define dataset
    help = """
  Demo code for multiple views and one person:

    - Input : {} => {}
    - Output: {}
    - Body  : {}
""".format(args.path, ', '.join(args.sub), args.out,
           args.body)
    print(help)
    from easymocap.dataset import MVMPMF

    dataset = MVMPMF(args.path, cams=args.sub, annot_root=args.annot,
                     config=CONFIG[args.body], kpts_type=args.body,
                     undis=args.undis, no_img=True, out=args.out, filter2d=cfg.dataset)
    mvposev1(dataset, args, cfg)