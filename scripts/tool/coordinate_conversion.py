'''
  @ Date: 2021/11/11 16:23
  @ Author: Zhao YaChen
'''

import numpy as np
import os
import json
import argparse
import time

mkdir = lambda x:os.makedirs(x, exist_ok=True)
mkout = lambda x:mkdir(os.path.dirname(x))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)

def load_Rt(extrixname):
    assert os.path.exists(extrixname), extrixname
    gp_paras = read_json(extrixname)
    R = np.array(gp_paras['R'])
    t = np.array(gp_paras['t'])
    return R, t[:, None]

def load_kp3d(jsonname):
    # mapname = {'frame', 'global_3dpose_f'}
    assert os.path.exists(jsonname), jsonname
    data = read_json(jsonname)
    if not isinstance(data, list):
        data = data['frames']
    for i in range((len(data))):  # sum of frame
        if 'frame' not in data[i].keys() or 'global_3dpose_f' not in data[i].keys():
            data.remove(data[i])
            continue
        global_3dpose_f = np.array(data[i]['global_3dpose_f'].split(" ")).astype('float64')
        if global_3dpose_f.size != 72:
            data.remove(data[i])
            continue
        kp3d_original = global_3dpose_f.reshape((24, 3))
        # data[i]['kp3d_original'] = kp3d_original
        # kp3d_conversion
        kp3d_conversion = convert_Qc2Qg(kp3d_original, R, t)
        data[i]['kp3d_conversion'] = ' '.join(str(item) for innerlist in kp3d_conversion for item in innerlist)
    data.sort(key=lambda x: x['frame'])
    result = {}
    result['frames'] = data
    return result

def convert_Qc2Qg(Qc, R, t):
    """
    根据旋转和平移参数将图像坐标系下的Qc(Xc,Yc,Zc)转变成世界坐标系下的Qg(Xg,Yg,Zg)
    """
    Qg = (Qc - t.T) @ R
    return Qg

def get_camera_center():
    # 定义内参矩阵K
    K = np.array([
        [-165.09018079638793, -137.76594686806902, 1077.9999999999998],
        [-2.940506569504228, -39.95582167976235, 536.0],
        [0.004300150429324309, -0.14513926534531338, 1.0]
    ], dtype=np.float64)
    invK = np.linalg.inv(K)
    # 定义外参矩阵R
    R = np.array([
        [-0.9992417025093056, -0.0309570493769164, 0.02361527175748146],
        [-0.0301018788899249, 0.2295179843853277, -0.9728388210443665],
        [0.024696089844362588, -0.9728119838579146, -0.23027580682483234]
    ], dtype=np.float64)
    # Rvec = cv2.Rodrigues(R)[0]
    R_T = R.T
    # 定义平移矩阵t
    t = np.array([0.8930804586971847, -0.20677865169745893, 7.0435064628030775])
    t_T = t.T
    t_T_None = t_T[:, None]
    # 定义Rt矩阵
    Rt = np.hstack((R, t_T_None))
    # T=(R,t)的逆矩阵应该是T_inverse=(R_transpose, -R_transpose * t)
    invRt = np.hstack((R_T, -R_T @ t_T_None))

    # 求相机中心：计算世界坐标系下相机中心坐标
    # Cc = [0, 0, 0]
    Cc = [0.142559058339787, -1.5699610434772517, 8.75408759158108]
    # 计算Cg：Cg = -R.T * t 或者 -R' * t
    Cg = - R.T @ t
    # 验证：Cc = R * Cg + t
    evalCg = R @ np.array(Cg)[:, None] + t_T_None

    # 第一种方案：Rt的逆乘上图像坐标系变成世界坐标系的坐标
    # 增广坐标homeCc
    homeCc = np.concatenate([Cc, [1]], axis=-1)
    homeCg = invRt @ homeCc

    # 第二种方案：从像素坐标系Q = [u, v, 1]转变到图像坐标系Qc再转变到Qg
    # Q = [u, v, 1]
    # Qc = Ki * Q 其中 Ki = inv(K)
    # Qg = R' * Qc -R' * t 或者 R' * (Qc - t) 或者 (Qc - t) * R

    # Qg = R' * (Ki * Q - t) 或者 (Q * Ki - t) * R
    pluckerCg = (Cc - t.T) @ R
    pluckerCg2 = R_T @ (Cc - t.T)

    print('Cg:', Cg)
    print('homeCg:', homeCg)
    print('pluckerCg:', pluckerCg)
    print('pluckerCg2:', pluckerCg2)

    return Cg




def load_parser():
    parser = argparse.ArgumentParser('Demo Code For Converting Coordinate')
    parser.add_argument('--from_file', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--extrix', type=str, default=None)
    return parser

def parse_parser(parser):
    args = parser.parse_args()
    if args.from_file is not None:
        assert os.path.exists(args.from_file), args.from_file
    if args.out is None:
        print(' - [Warning] Please specify the output path `--out ${out}`')
        args.out = os.getcwd()
        print(' - [Warning] Default to {}'.format(args.out))
    if args.extrix is None:
        print(' - [Warning] Please specify the extrix path `--extrix ${extrix}`')
    return args

if __name__ == '__main__':
    datetime = time.strftime("%Y%m%d%H%M%S")
    parser = load_parser()
    args = parse_parser(parser)
    out_name = os.path.join(args.out, 'kp3d_conversion_' + datetime + '.json')
    help = """
  Demo code for converting coordinate:
    - Input : {} 
    - Output: {}
    - Extrix: {}
""".format(args.from_file, args.out, args.extrix)
    print(help)
    # 第一步：通过gp_parse.json读取R外参矩阵和t内参矩阵
    R, t = load_Rt(args.extrix)
    # 第二步：将kp3d.json中的global_3dpose_f{list:72}转变成世界坐标系的坐标
    frames = load_kp3d(args.from_file)
    # 第三步：保存转换后的坐标到kp3d_conversion.json中
    save_json(out_name, frames)
    print('Convert Finished!')
