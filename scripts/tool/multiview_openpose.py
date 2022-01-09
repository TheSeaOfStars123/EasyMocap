'''
  @ Date: 2022/1/4 15:09
  @ Author: Zhao YaChen
'''
from tqdm import tqdm
import os
from os.path import join
from glob import glob
from threading import Thread
import socket
import cv2
import numpy as np
from read_pose_heatmaps import load_coords_net_resolution_by_person, estimate_pose_by_given_part, save_txt_for_detection
import sys
from sys import platform


extensions = ['.mp4', '.avi']

def run(cmd):
    print(cmd)
    os.chdir("D:/Downloads/ffmpeg-4.3.2-essentials_build/ffmpeg-4.3.2-2021-02-27-essentials_build/bin")
    os.system(cmd)

def extract_images(path, ffmpeg, image, start_number, vframes):
    '''
    extract image from videos
    '''
    videos = sorted(sum([
        glob(join(path, 'videos', '*'+ext)) for ext in extensions
        ], [])
    )
    subs = []
    subnames = []
    for videoname in videos:
        sub = '.'.join(os.path.basename(videoname).split('.')[:-1])
        sub = sub.replace(args.strip, '')
        subs.append(sub)
        outpath = join(path, image, sub)
        os.makedirs(outpath, exist_ok=True)
        subname = '{:06d}'.format(start_number)  # '000000'
        subnames.append(subname)
        other_cmd = ''
        if args.num != -1:
            other_cmd += '-vframes {}'.format(vframes)
        if args.transpose != -1:
            other_cmd += '-vf transpose={}'.format(args.transpose)
        cmd = '{} -i {} {} -q:v 1 -start_number {} -y {}.jpg'.format(
            ffmpeg, videoname, other_cmd, start_number, join(outpath, subname))
        run(cmd)
    return subs, subnames

def extract_2d(image, keypoints, heatmaps):
    skip = False
    if not skip:
        # os.makedirs(keypoints, exist_ok=True)
        # os.makedirs(heatmaps, exist_ok=True)
        # Import Openpose (Windows/Ubuntu/OSX)
        # dir_path = os.path.dirname(os.path.realpath(__file__))

        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread(image)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # # Process outputs
        # outputImageF = (datum.inputNetData[0].copy())[0, :, :, :] + 0.5
        # outputImageF = cv2.merge([outputImageF[0, :, :], outputImageF[1, :, :], outputImageF[2, :, :]])
        # outputImageF = (outputImageF * 255.).astype(dtype='uint8')
        keypoint = datum.poseKeypoints  # keypoints(personNum, 25, 3)
        heatmaps = datum.poseHeatMaps.copy()  # heatmaps(77, 368, 368)
        # heatmaps = (heatmaps).astype(dtype='uint8')
        pafMat = datum.poseHeatMaps[25:, :, :]
        # keypoints_Frame.append(keypoints)
        # pafMat_Frame.append(pafMat)
        # # Display Image
        # counter = 0
        # while 1:
        #     num_maps = heatmaps.shape[0]
        #     heatmap = heatmaps[counter, :, :].copy()
        #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     combined = cv2.addWeighted(outputImageF, 0.5, heatmap, 0.5, 0)
        #     cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", combined)
        #     key = cv2.waitKey(-1)
        #     if key == 27:
        #         break
        #     counter += 1
        #     counter = counter % num_maps
        return keypoint, pafMat


def extract_2d_thread(image, keypoints, heatmaps):
    thread = Thread(target=extract_2d, args=(image, keypoints, heatmaps))
    thread.start()
    return thread

def multiview_openpose(path, ffmpeg, image, num):
    start = 500
    keypoints_Frame = []
    pafMat_Frame = []
    for nf in tqdm(range(start, start + num)):
        keypoints_Frame = []
        pafMat_Frame = []
        # subs, subnames = extract_images(path, ffmpeg, image, nf, 1)
        subs = ['0', '1', '2', '3', '4', '5']
        subname = '{:06d}'.format(nf)  # '000000'
        # subnames = ['000000', '000000', '000000', '000000', '000000', '000000']
        # for sub, subname in zip(subs, subnames):
        for sub in subs:
            global_tasks = []
            image_root = join(args.path, 'images', sub, subname+'.jpg')
            annot_root = join(args.path, args.annot, sub, subname+'.jpg')
            keypoints_root = join(args.path, args.openpose_part_candidates, sub, subname+'.json')
            heatmap_root = join(args.path, args.output_heatmaps_folder, sub, subname+'.float')
            keypoints, heatmaps = extract_2d(image_root, keypoints_root, heatmap_root)
            # global_tasks.append(extract_2d_thread(image_root, keypoints_root, heatmap_root))
            keypoints_Frame.append(keypoints)
            pafMat_Frame.append(heatmaps)
        # for task in global_tasks:
        #     task.join()

        out_txt_root = join(args.path, args.OUT_TXT, str(nf) + '.txt')
        for sub, keypoint, pafMat in zip(subs, keypoints_Frame, pafMat_Frame):
            origin_coords, coords = load_coords_net_resolution_by_person(keypoint)
            origin_connection_all, origin_coords = estimate_pose_by_given_part(origin_coords, coords, pafMat)
            with open(out_txt_root, 'a') as f:  # 设置文件对象
                # f.write('4\t300\n')  # 将字符串写入文件中
                save_txt_for_detection(origin_coords, origin_connection_all, f, 1)
        f.close()
        data = str.encode(str(nf) + '\n')  # 发送当前帧数
        sock.send(data)
    # 结束标志：发送一个Q
    sock.send(str.encode('Q'))
    # 4. 关闭套接字，关闭加载的套接字库(closesocket()/WSACleanup())
    sock.close()
    cv2.destroyAllWindows()


net_resolution_X = 368
net_resolution_Y = 368


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--strip', type=str, default='')
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--transpose', type=int, default=-1)
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg')

    parser.add_argument('--annot', type=str, default='annots',
                        help="sub directory name to store the generated annotation files, default to be annots")
    parser.add_argument('--openpose_part_candidates', type=str, default='openpose_part_candidates')
    parser.add_argument('--output_heatmaps_folder', type=str, default='output_heatmaps_folder')
    parser.add_argument('--OUT_TXT', type=str, default='detections')

    parser.add_argument('--mode', type=str, default='4d_association', choices=[
        '4d_association', 'easymocap'], help="model to extract 3d joints from videos")
    parser.add_argument('--openpose', type=str, default='D:\Desktop\MOCAP\openpose')
    parser.add_argument('--ext', type=str, default='.jpg')
    # parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    # 1. 加载套接字库，创建套接字(WSAStartup()/socket())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 2. 向服务器发出连接请求(connect())
    address_server = ('127.0.0.1', 8010)
    sock.connect(address_server)

    # 启动openpose
    dir_path = "D:\\Desktop\\openpose\\build\\examples\\tutorial_api_python"
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release')
            os.environ['PATH'] = os.environ[
                                     'PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
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
    params["keypoint_scale"] = 1
    params["model_folder"] = "D:\Desktop\openpose\models"
    params["heatmaps_add_parts"] = True
    # params["heatmaps_add_bkg"] = True
    params["heatmaps_add_PAFs"] = True
    params["heatmaps_scale"] = 0

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    multiview_openpose(args.path, args.ffmpeg, args.image, args.num)

    # 结束标志：发送一个Q
    sock.send(str.encode('Q'))
    # 4. 关闭套接字，关闭加载的套接字库(closesocket()/WSACleanup())
    sock.close()
    cv2.destroyAllWindows()
