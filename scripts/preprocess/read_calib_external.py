from numpy import loadtxt
import numpy as np
import scipy.io as scio
import cv2

def read_shelf_calib():
    """
    对于shelf数据集中给定的txt参数文件进行读取
    并转化为本项目中可以使用的内参外参文件
    """
    shelf_mat_path = "D:\Desktop\MOCAP\EasyMocap-master\config\evaluation\prjectionMat_shelf.mat"
    projection_mat = scio.loadmat(shelf_mat_path)
    # load data from txt-file
    N = 5
    cams = {}
    for i in range(N):
        txt_path = "D:/Desktop/MOCAP/EasyMocap-master/0_input/20211105_Shelf/Camera%d.txt" % i
        cam = str(i)
        # p
        P_data = loadtxt(txt_path, skiprows=0, max_rows=3)  # 前三行是投影矩阵P
        # K R T
        KRT_data = loadtxt(txt_path, skiprows=4)
        cams[cam] = {}
        cams[cam]['K'] = KRT_data[0:3, 0:3]  # 第5-7行是内参矩阵K
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        cams[cam]['R'] = KRT_data[3:6, 0:3]  # 第8-10行是旋转矩阵
        cams[cam]['T'] = KRT_data[6:, 0:3].T  # 第11行是平移矩阵
        cams[cam]['RT'] = np.hstack((cams[cam]['R'], cams[cam]['T']))
        cams[cam]['P'] = P_data
        # 离谱：为什么P不等于KRT
        cams[cam]['Pcomp'] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['dist'] = np.zeros((1, 5))
    # write
    from easymocap.mytools import write_camera
    write_camera(cams, 'D:/Desktop/MOCAP/EasyMocap-master/0_input/20211105_Shelf')


def read_campus_calib():
    """
    对于campus数据集中给定的参数
    转化为本项目中可以使用的内参外参文件
    """
    campus_mat_path = "D:\Desktop\MOCAP\EasyMocap-master\config\evaluation\prjectionMat_campus.mat"
    projection_mat = scio.loadmat(campus_mat_path)
    # load data from txt-file
    N = 3
    cams = {}
    for i in range(N):
        txt_path = "D:/Desktop/MOCAP/EasyMocap-master/0_input/20211106_Campus/calib/Camera%d.txt" % i
        cam = str(i)
        # p
        P_data = loadtxt(txt_path, skiprows=0, max_rows=3)  # 前三行是投影矩阵P
        # K Rvec Tvec
        KRT_data = loadtxt(txt_path, skiprows=3, max_rows=5)
        # RT
        RT_data = loadtxt(txt_path, skiprows=8, max_rows=3)
        cams[cam] = {}
        cams[cam]['K'] = KRT_data[0:3, 0:3]  # 第4-6行是内参矩阵K
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = KRT_data[3:4, 0:3]  # 第7行是旋转向量
        Tvec = KRT_data[4:5, 0:3].T  # 第8行是平移向量
        R = cv2.Rodrigues(Rvec)[0]
        RTcomp = np.hstack((R, Tvec))
        cams[cam]['RTcomp'] = RTcomp
        # 第9-11行是RT矩阵
        cams[cam]['R'] = RT_data[0:3, 0:3]
        cams[cam]['T'] = Tvec
        cams[cam]['RT'] = RT_data
        cams[cam]['P'] = P_data
        # 离谱：为什么P不等于KRT
        cams[cam]['Pcomp'] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['dist'] = np.zeros((1, 5))
    # write
    from easymocap.mytools import write_camera
    write_camera(cams, 'D:/Desktop/MOCAP/EasyMocap-master/0_input/20211106_Campus')


if __name__ == '__main__':
    # read_shelf_calib()
    read_campus_calib()
