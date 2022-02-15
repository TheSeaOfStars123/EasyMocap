'''
  @ Date: 2022/1/4 15:09
  @ Author: Zhao YaChen
'''
import subprocess
import os
from os.path import join
from glob import glob
import socket
import cv2
import numpy as np
from read_pose_heatmaps import load_coords_net_resolution_by_person, estimate_pose_by_given_part, save_txt_for_detection
import sys
from sys import platform
import math
import shutil

extensions = ['.mp4', '.avi']

# 定义函数，第一个参数是缩放比例，第二个参数是需要显示的图片组成的元组或者列表
def ManyImgs(scale, imgarray):
    rows = len(imgarray)         # 元组或者列表的长度
    cols = len(imgarray[0])      # 如果imgarray是列表，返回列表里第一幅图像的通道数，如果是元组，返回元组里包含的第一个列表的长度
    # print("rows=", rows, "cols=", cols)

    # 判断imgarray[0]的类型是否是list
    # 是list，表明imgarray是一个元组，需要垂直显示
    rowsAvailable = isinstance(imgarray[0], list)

    # 第一张图片的宽高
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    # print("width=", width, "height=", height)

    # 如果传入的是一个元组
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # 遍历元组，如果是第一幅图像，不做变换
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                # 将其他矩阵变换为与第一幅图像相同大小，缩放比例为scale
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)
                # 如果图像是灰度图，将其转换成彩色显示
                if  len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        # 创建一个空白画布，与第一张图片大小相同
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows   # 与第一张图片大小相同，与元组包含列表数相同的水平空白图像
        for x in range(0, rows):
            # 将元组里第x个列表水平排列
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)   # 将不同列表垂直拼接
    # 如果传入的是一个列表
    else:
        # 变换操作，与前面相同
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # 将列表水平排列
        hor = np.hstack(imgarray)
        ver = hor
    return ver


def display(datums, num_row):
    imgs = []
    for datum in datums:
        imgs.append(datum.cvOutputData)
    # 按照两行进行展示
    stackedimageb = ManyImgs(0.2, (imgs[0:num_row], imgs[num_row:]))
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", stackedimageb)
    key = cv2.waitKey(1)
    return (key == 27)


def multiview_openpose(datums, subs, nf):
    keypoints_Frame = []
    pafMat_Frame = []
    for datum in datums:
        keypoint = datum.poseKeypoints  # keypoint(personNum, 25, 3)
        pafMat = datum.poseHeatMaps[25:, :, :]  # pafMat(52, net_resolution_X, net_resolution_Y)
        keypoints_Frame.append(keypoint)
        pafMat_Frame.append(pafMat)
    out_txt_root = join(detection_path, str(nf) + '.txt')
    # s = ""
    for sub, keypoint, pafMat in zip(subs, keypoints_Frame, pafMat_Frame):
        origin_coords, coords = load_coords_net_resolution_by_person(keypoint, pafMat.shape[1], pafMat.shape[2])
        origin_connection_all, origin_coords = estimate_pose_by_given_part(origin_coords, coords, pafMat)
        with open(out_txt_root, 'a') as f:  # 设置文件对象
            save_txt_for_detection(origin_coords, origin_connection_all, f, 1)
            # s = save_txt_for_detection(origin_coords, origin_connection_all, s, 1)
    f.close()
    data = str.encode(str(nf) + '\n')  # 发送当前帧数
    # data = str.encode(str(len(s)) + '\n')
    sock.send(data)
    # sock.send(str.encode(s))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--ext', type=str, default='.avi')
    parser.add_argument('--skel_path', type=str, default='D:/Desktop/4D_ASSOCIATION/4d_association_data/skel/SKEL19')
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--openpose_demo', type=str, default='D:/Desktop/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose/python')
    parser.add_argument('--vis_server_demo', type=str, default='python ../../apps/vis/vis_server.py --cfg ../../config/vis3d/o3d_scene.yml write True camera.cz 3. camera.cy 0.5')
    parser.add_argument('--association_demo', type=str, default='D:/Desktop/4D_ASSOCIATION/4d_association/x64/Release/mocap.exe')
    parser.add_argument('--net_resolution', type=str, default='-1x224')
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = args.path
    detection_path = join(args.path, 'detections')
    keypoints3d_path = join(args.out_path, 'keypoints3d')
    vis3d_path = join(args.out_path, 'vis3d')
    # 判断标定文件是否存在
    assert os.path.exists(join(args.path, 'calibration.json')), join(args.path, 'calibration.json')
    # 判断skel_path是否存在
    assert os.path.exists(args.skel_path), args.skel_path
    # 创建最新detections文件夹
    if os.path.exists(detection_path):
        shutil.rmtree(detection_path)
        os.makedirs(detection_path)
    else:
        os.makedirs(detection_path)
    # 创建keypoints文件夹
    if not os.path.exists(keypoints3d_path):
        os.makedirs(keypoints3d_path)
    # 创建vis3d文件夹
    if not os.path.exists(vis3d_path):
        os.makedirs(vis3d_path)

    vis_server = subprocess.Popen(args.vis_server_demo + " out " + vis3d_path, shell=True)
    association = subprocess.Popen(args.association_demo + " " + args.path + " " + args.skel_path + " " + args.out_path
                                   + " " + args.ext, shell=True)

    # 1. 加载套接字库，创建套接字(WSAStartup()/socket())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 2. 向服务器发出连接请求(connect())
    address_server = ('127.0.0.1', 8010)
    sock.connect(address_server)

    try:
        dir_path = args.openpose_demo
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../bin/python/openpose/Release')
                os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' + dir_path + '/../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = join(dir_path, "../models")
        params["keypoint_scale"] = 1
        params["heatmaps_add_parts"] = True
        params["heatmaps_add_PAFs"] = True
        params["heatmaps_scale"] = 0

        # 对每个视频开启OpenPose进程
        # 罗列path中的视频
        if os.path.isdir(args.path):
            subs = []
            opWrappers = []
            videos = sorted(sum([glob(join(args.path, 'video', '*' + ext)) for ext in extensions], []))
            if len(videos) > 0:
                for videoname in videos:
                    basename = '.'.join(os.path.basename(videoname).split('.')[:-1])
                    subs.append(basename)
                    params = dict()
                    params["model_folder"] = join(dir_path, "../models")
                    params["video"] = videoname
                    params["keypoint_scale"] = 1
                    params["heatmaps_add_parts"] = True
                    params["heatmaps_add_PAFs"] = True
                    params["heatmaps_scale"] = 0
                    params["net_resolution"] = args.net_resolution
                    # Starting OpenPose(AsynchronousOut)
                    opWrapper = op.WrapperPython(op.ThreadManagerMode.AsynchronousOut)
                    opWrapper.configure(params)
                    opWrappers.append(opWrapper)
            else:
                print('videos文件夹中没有符合要求的视频！')
                sock.close()
                cv2.destroyAllWindows()
                vis_server.terminate()
                association.terminate()
                sys.exit(0)

            print('cameras: ', ' '.join(subs))
            for opWrapper in opWrappers:
                opWrapper.start()

            # Main loop
            userWantsToExit = False
            nf = -1
            while not userWantsToExit:
                ignoreFrame = False
                nf += 1
                datumProcesseds = op.VectorDatum()
                for opWrapper in opWrappers:
                    # Pop frame
                    datumProcessed = op.VectorDatum()
                    if opWrapper.waitAndPop(datumProcessed):
                        if datumProcessed[0].poseKeypoints is not None:
                            datumProcesseds.append(datumProcessed[0])
                        else:
                            ignoreFrame = True
                            continue
                    else:
                        break
                if not ignoreFrame:
                    userWantsToExit = display(datumProcesseds, num_row=math.ceil(len(subs) // 2))
                    multiview_openpose(datumProcesseds, subs, nf)
                else:
                    nf -= 1
            # 结束标志：发送一个Q
            sock.send(str.encode('Q'))
            # 4. 关闭套接字，关闭加载的套接字库(closesocket()/WSACleanup())
            sock.close()
            cv2.destroyAllWindows()
            sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(-1)