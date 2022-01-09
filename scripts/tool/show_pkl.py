'''
  @ Date: 2021/12/17 14:46
  @ Author: Zhao YaChen
'''
# show_pkl.py

import pickle
import os.path as osp


def load_bodydata(model_type, model_path, gender):
    if osp.isdir(model_path):
        model_fn = '{}_{}.{ext}'.format(model_type.upper(), gender.upper(), ext='pkl')
        smpl_path = osp.join(model_path, model_fn)
    else:
        smpl_path = model_path
    assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
        smpl_path)

    with open(smpl_path, 'rb') as smpl_file:
        data = pickle.load(smpl_file, encoding='latin1')
    return data

import json
import numpy as np
def load_json(datas):
    if isinstance(datas, str):
        datas = json.loads(datas)
    for data in datas:
        for key in data.keys():
            if key == 'id':
                continue
            data[key] = np.array(data[key])

# data = load_bodydata('smpl', 'D:\Desktop\MOCAP\EasyMocap-master\data\smplx\smpl', 'male')
# print(data)
# print(len(data))

data = '   [{"id":0,"keypoints3d":[[-0.33600000000000002,1.014,1.4950000000000001,0.85099999999999998],[-0.42199999999999999,1.1719999999999999,1.3,0.88200000000000001],[-0.55600000000000005,1.123,1.2889999999999999,0.83799999999999997],[-0.63600000000000001,1.135,1.038,0.84099999999999997],[-0.67700000000000005,1.069,0.82299999999999995,0.86199999999999999],[-0.28100000000000003,1.216,1.3069999999999999,0.85399999999999998],[-0.21099999999999999,1.254,1.0800000000000001,0.83199999999999996],[-0.14000000000000001,1.2090000000000001,0.86499999999999999,0.86899999999999999],[-0.38800000000000001,1.1479999999999999,0.78200000000000003,0.76300000000000001],[-0.48299999999999998,1.1240000000000001,0.78400000000000003,0.75800000000000001],[-0.48099999999999998,1.153,0.42799999999999999,0.81100000000000005],[-0.48599999999999999,1.1539999999999999,0.078,0.83599999999999997],[-0.28699999999999998,1.1799999999999999,0.78000000000000003,0.74299999999999999],[-0.23499999999999999,1.1719999999999999,0.44600000000000001,0.78600000000000003],[-0.17599999999999999,1.2150000000000001,0.089999999999999997,0.80200000000000005],[-0.38500000000000001,1.022,1.524,0.88600000000000001],[-0.31900000000000001,1.0589999999999999,1.5169999999999999,0.79200000000000004],[-0.47699999999999998,1.0740000000000001,1.518,0.82999999999999996],[-0.33500000000000002,1.1539999999999999,1.504,0.81899999999999995],[-0.058000000000000003,1.121,0.050000000000000003,0.72599999999999998],[-0.057000000000000002,1.167,0.058000000000000003,0.76500000000000001],[-0.193,1.242,0.050999999999999997,0.76900000000000002],[-0.47299999999999998,1.008,0.042000000000000003,0.71499999999999997],[-0.50800000000000001,1.0309999999999999,0.041000000000000002,0.70399999999999996],[-0.47499999999999998,1.1930000000000001,0.040000000000000001,0.78200000000000003]]},{"id":1,"keypoints3d":[[-1.1160000000000001,-1.0820000000000001,1.4930000000000001,0.83599999999999997],[-1.1930000000000001,-1.234,1.3420000000000001,0.86699999999999999],[-1.071,-1.3320000000000001,1.357,0.84599999999999997],[-1.0549999999999999,-1.4450000000000001,1.099,0.81200000000000006],[-1.1220000000000001,-1.423,0.876,0.747],[-1.3180000000000001,-1.1399999999999999,1.333,0.84999999999999998],[-1.373,-1.1739999999999999,1.0780000000000001,0.72999999999999998],[-1.3340000000000001,-1.321,0.86599999999999999,0.65400000000000003],[-1.1519999999999999,-1.2769999999999999,0.82399999999999995,0.76600000000000001],[-1.071,-1.335,0.82999999999999996,0.755],[-1.036,-1.3520000000000001,0.46200000000000002,0.80700000000000005],[-0.997,-1.3899999999999999,0.109,0.75800000000000001],[-1.2430000000000001,-1.2210000000000001,0.81599999999999995,0.749],[-1.2569999999999999,-1.26,0.45000000000000001,0.81999999999999995],[-1.27,-1.302,0.079000000000000001,0.79100000000000004],[-1.099,-1.1040000000000001,1.5269999999999999,0.79700000000000004],[-1.1659999999999999,-1.069,1.5249999999999999,0.81100000000000005],[-1.1040000000000001,-1.2030000000000001,1.5409999999999999,0.83399999999999996],[-1.2509999999999999,-1.109,1.5189999999999999,0.87],[-1.208,-1.159,0.036999999999999998,0.71799999999999997],[-1.264,-1.1659999999999999,0.043999999999999997,0.73999999999999999],[-1.2809999999999999,-1.339,0.029999999999999999,0.69399999999999995],[-0.89100000000000001,-1.298,0.065000000000000002,0.69199999999999995],[-0.877,-1.3280000000000001,0.071999999999999995,0.66600000000000004],[-1.016,-1.413,0.062,0.749]]},{"id":2,"keypoints3d":[[-0.084000000000000005,-0.16400000000000001,1.016,0.873],[-0.24099999999999999,-0.032000000000000001,0.91300000000000003,0.84699999999999998],[-0.35799999999999998,-0.14000000000000001,0.91000000000000003,0.77900000000000003],[-0.57999999999999996,-0.154,0.76500000000000001,0.78900000000000003],[-0.40600000000000003,-0.20699999999999999,0.64300000000000002,0.68600000000000005],[-0.129,0.074999999999999997,0.92400000000000004,0.81599999999999995],[-0.13500000000000001,0.32500000000000001,0.80300000000000005,0.80200000000000005],[-0.056000000000000001,0.20999999999999999,0.66100000000000003,0.79000000000000004],[-0.26200000000000001,0.048000000000000001,0.46100000000000002,0.65500000000000003],[-0.34200000000000003,-0.029000000000000001,0.44600000000000001,0.61399999999999999],[-0.074999999999999997,-0.375,0.48199999999999998,0.755],[-0.113,-0.371,0.080000000000000002,0.77200000000000002],[-0.17000000000000001,0.125,0.47599999999999998,0.67400000000000004],[0.17399999999999999,-0.14499999999999999,0.51100000000000001,0.80700000000000005],[0.20799999999999999,-0.0050000000000000001,0.090999999999999998,0.74099999999999999],[-0.127,-0.19700000000000001,1.054,0.69799999999999995],[-0.075999999999999998,-0.152,1.0469999999999999,0.73999999999999999],[-0.22600000000000001,-0.22900000000000001,1.0629999999999999,0.72099999999999997],[-0.104,-0.064000000000000001,1.083,0.82699999999999996],[0.36399999999999999,-0.115,0.062,0.80100000000000005],[0.36399999999999999,-0.052999999999999999,0.052999999999999999,0.81000000000000005],[0.182,0.042000000000000003,0.047,0.65500000000000003],[-0.012999999999999999,-0.497,0.037999999999999999,0.65600000000000003],[-0.058000000000000003,-0.5,0.034000000000000002,0.68799999999999994],[-0.14799999999999999,-0.34000000000000002,0.048000000000000001,0.70499999999999996]]},{"id":3,"keypoints3d":[[1.0629999999999999,-1.3169999999999999,1.5960000000000001,0.90800000000000003],[1.125,-1.3779999999999999,1.3919999999999999,0.89800000000000002],[1.2310000000000001,-1.268,1.3979999999999999,0.86099999999999999],[1.141,-1.155,1.194,0.77000000000000002],[0.97099999999999997,-1.339,1.198,0.76300000000000001],[0.99399999999999999,-1.496,1.387,0.878],[0.90900000000000003,-1.4370000000000001,1.1479999999999999,0.879],[1.0669999999999999,-1.2290000000000001,1.1830000000000001,0.80900000000000005],[1.085,-1.3480000000000001,0.86199999999999999,0.81599999999999995],[1.143,-1.264,0.87,0.78700000000000003],[1.1859999999999999,-1.3069999999999999,0.47199999999999998,0.84399999999999997],[1.2430000000000001,-1.355,0.090999999999999998,0.77200000000000002],[1.018,-1.4279999999999999,0.86399999999999999,0.81000000000000005],[1.0389999999999999,-1.4319999999999999,0.46999999999999997,0.81200000000000006],[1.081,-1.4770000000000001,0.108,0.84499999999999997],[1.0900000000000001,-1.327,1.6299999999999999,0.93999999999999995],[1.0269999999999999,-1.3600000000000001,1.6279999999999999,0.88600000000000001],[1.175,-1.3220000000000001,1.615,0.74199999999999999],[1.0389999999999999,-1.429,1.583,0.747],[0.98699999999999999,-1.371,0.062,0.71499999999999997],[0.96699999999999997,-1.4350000000000001,0.069000000000000006,0.72399999999999998],[1.1160000000000001,-1.4890000000000001,0.058000000000000003,0.70599999999999996],[1.1279999999999999,-1.264,0.042999999999999997,0.70899999999999996],[1.1990000000000001,-1.238,0.042000000000000003,0.71699999999999997],[1.254,-1.385,0.051999999999999998,0.68999999999999995]]}]'
load_json(data)