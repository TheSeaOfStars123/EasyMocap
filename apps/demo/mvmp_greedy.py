# @Time : 2021/11/14 9:11 PM
# @Author : zyc
# @File : mvmp_greedy.py
# @Title :
# @Description :
from easymocap.dataset import CONFIG
from easymocap.affinity.affinity import ComposedAffinity
from easymocap.assignment.associate import simple_associate
from easymocap.assignment.group import PeopleGroup

from easymocap.mytools import Timer
from tqdm import tqdm


def greedy_graph(result, affinity, dimGroups, nf):
    import numpy as np
    maxid = 0
    frames_of_id = {}
    log = print
    log = lambda x: x
    pemi = np.zeros((affinity.shape[0], affinity.shape[1]))
    views = np.zeros(dimGroups[-1], dtype=np.int)
    for nv in range(len(dimGroups) - 1): # dimGroups:[0,4,7,11...]
        views[dimGroups[nv]:dimGroups[nv + 1]] = nv # views:[0,0,0,0,1,1,1,2,2,2,2...]
    # 第一步：按照分数从大到小排序
    # 第二步：如果候选edge对着的两条边的指向都没有id,那么就建立连接
    # 第三步：已知1和5建立连接
    #   想建立5和10的连接
    #       如果10没有连接 那么只需要考虑1和10之间的分数是不是有效连接，如果不是就不能建立连接。如果是可以建立连接
    #       如果10有连接 并且对应的是另一个人 那么5和10就不能建立连接。如果对应的是同一个人 可以建立连接
    is_valid_affinity = affinity
    is_valid_affinity[pemi == 1] = 0
    while np.max(is_valid_affinity) > 0.5:
        edge_candidate = np.where(is_valid_affinity == np.max(is_valid_affinity))
        nf0 = views[edge_candidate[0][0]]
        ni0 = edge_candidate[0][0] - dimGroups[nf0]
        nf1 = views[edge_candidate[0][1]]
        ni1 = edge_candidate[0][1] - dimGroups[nf1]
        # 先看有没有连接
        id0 = result[nf0][ni0]['id']
        id1 = result[nf1][ni1]['id']
        # directly assign
        if id0 == -1 and id1 == -1:
            result[nf0][ni0]['id'] = maxid
            pemi[edge_candidate[0][0]][edge_candidate[0][1]] = 1
            pemi[edge_candidate[0][1]][edge_candidate[0][0]] = 1
            log('Create person {}'.format(maxid))
            frames_of_id[maxid] = {nf0: ni0, nf1: ni1}
            maxid += 1
        elif id0 != -1 and id1 == -1:
            # 5---10   27 10和27想要连接，判断27和5之间是否有效
            # if(affinity[edge_candidate[0][0]][edge_candidate[0][1]] > 0.5):
            # 还要判断id0中是否已经与当前视图nf1中其他人次对应：
            if nf1 in frames_of_id[id0].keys():
                log('Merge conflict')
            result[nf1][ni1]['id'] = id0
            # log('Merge person {}'.format(maxid))
            frames_of_id[id0][nf1] = ni1
            pemi[edge_candidate[0][0]][edge_candidate[0][1]] = 1
            pemi[edge_candidate[0][1]][edge_candidate[0][0]] = 1
        elif id0 == -1 and id1 != -1:
            # 0   5---10 0和5想要连接，判断0和10之间是否有效
            # if (affinity[edge_candidate[0][0]][edge_candidate[0][1]] > 0.5):
            # 还要判断id1中是否已经与当前视图nf0中其他人次对应：
            if nf0 in frames_of_id[id1].keys():
                log('Merge conflict')
            result[nf0][ni0]['id'] = id1
            frames_of_id[id1][nf0] = ni0
            pemi[edge_candidate[0][0]][edge_candidate[0][1]] = 1
            pemi[edge_candidate[0][1]][edge_candidate[0][0]] = 1
        elif id0 == id1 and id0 != -1:
            pemi[edge_candidate[0][0]][edge_candidate[0][1]] = 1
            pemi[edge_candidate[0][1]][edge_candidate[0][0]] = 1
        # merge
        elif id0 != id1:
            common = frames_of_id[id0].keys() & frames_of_id[id1].keys()
            for key in common:  # conflict
                if frames_of_id[id0][key] == frames_of_id[id1][key]:
                    pass
                else:
                    is_valid_affinity[edge_candidate[0][0]][edge_candidate[0][1]] = 0
                    is_valid_affinity[edge_candidate[0][1]][edge_candidate[0][0]] = 0
                    break
            else:  # merge
                log('Merge {} to {}'.format(id1, id0))
                for key in frames_of_id[id1].keys():
                    result[key][frames_of_id[id1][key]]['id'] = id0
                    frames_of_id[id0][key] = frames_of_id[id1][key]
                    pemi[edge_candidate[0][0]][edge_candidate[0][1]] = 1
                    pemi[edge_candidate[0][1]][edge_candidate[0][0]] = 1
                frames_of_id.pop(id1)
                continue
            log('Conflict; not merged')
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
            import os
            # from ..mytools.camera_utils import read_camera, get_fundamental_matrix, Undistort
            # from ..mytools import FileWriter, read_annot, getFileList, save_json
            from easymocap.mytools import read_annot
            import cv2
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
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        # 计算不同视角的检测结果的affinity
        with Timer('compute affinity', not args.time):
            affinity, dimGroups = affinity_model(annots, images=images)
        # 根据Real-time论文中的贪心算法计算置换矩阵
        pemi, result = greedy_graph(result, affinity, dimGroups, nf)
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
    help="""
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