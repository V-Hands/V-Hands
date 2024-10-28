import json
import glob
from zipfile import ZipFile
import cv2
import numpy as np
from utils import *
import math
from utils import load_config
cfg = load_config()

folder = f'exp/{cfg.exp}/ik_results/'
gt_root = cfg.test_root

output_root = folder
start_frame = 0
MAX_EPE = 400
MAX_Half = MAX_EPE / math.sqrt(2)

def loadjs(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_names(folder):
    paths = glob.glob(folder + '/*')
    names = [path.split('/')[-1].split('\\')[-1].split('.')[0] for path in paths]
    return names

def load_frames(folder):
    frames = []
    for i in range(len(glob(f'{folder}/*'))):
        img = cv2.imread(f'{folder}/{i}.png')[:,:,:1]
        frames.append(img)
    return frames

def cal_EPE(joint, gt):
    dis = joint - gt
    dis2 = dis*dis
    Euclidean_error = dis2.sum(1).sqrt().mean()
    return float(Euclidean_error) * 1000

def check_on_screen(xyz, frame, x_M=0.35000655987354695, y_M=0.19737224635111683, W=71, H=41, z_m = -0.02):
    x, y, z = xyz
    if z < z_m:
        return False
    
    if len(frame.shape) > 2:
        frame = frame[:, :, 0]
    
    x_p = int(x / x_M * W - 0.5)
    y_p = int(y / y_M * H - 0.5)

    ds = [-2, -1, 0, 1, 2]
    for dx in ds:
        for dy in ds:
            cur_x = x_p + dx
            cur_y = y_p + dy
            if cur_y >= H:
                continue
            if cur_y < 0:
                continue
            if cur_x >= W:
                continue
            if cur_x < 0:
                continue
            if frame[cur_y][cur_x] > 170:
                return True
    return False

def cal_EPE_v(joint, gt, frame):
    dis = joint - gt
    dis2 = dis*dis
    Euclidean_error_list = dis2.sum(1).sqrt()

    result = 0
    count = 0
    
    for finger_id in range(0, 5):
        if check_on_screen(gt[4*finger_id + 4], frame):
            result += Euclidean_error_list[4*finger_id+1: 4*finger_id+5].mean()
            count += 1

    return count > 0, (float(result / count * 1000) if count > 0 else None)

def cal_EPE_xy_v(joint, gt, frame):
    dis = joint - gt
    dis = dis[:, :2]
    dis2 = dis*dis
    Euclidean_error_list = dis2.sum(1).sqrt()

    result = 0
    count = 0
    
    for finger_id in range(0, 5):
        if check_on_screen(gt[4*finger_id + 4], frame):
            result += Euclidean_error_list[4*finger_id+1: 4*finger_id+5].mean()
            count += 1

    return (float(result / count * 1000) if count > 0 else None)

def cal_EPE_z_v(joint, gt, frame):
    dis = joint - gt
    dis = dis[:, 2:]
    dis2 = dis*dis
    Euclidean_error_list = dis2.sum(1).sqrt()
    result = 0
    count = 0
    for finger_id in range(0, 5):
        if check_on_screen(gt[4*finger_id + 4], frame):
            result += Euclidean_error_list[4*finger_id+1: 4*finger_id+5].mean()
            count += 1
    return (float(result / count * 1000) if count > 0 else None)

def cal_EPE_xy(joint, gt):
    dis = joint - gt
    dis = dis[:, :2]
    dis2 = dis*dis
    Euclidean_error = dis2.sum(1).sqrt().mean()
    return float(Euclidean_error) * 1000

def cal_EPE_z(joint, gt):
    dis = joint - gt
    dis = dis[:, 2:]
    dis2 = dis*dis
    Euclidean_error = dis2.sum(1).sqrt().mean()
    return float(Euclidean_error) * 1000

def new_gt_to_old_gt(gts):
    hand_types = list(gts.keys())
    old_gts = []
    for i in range(len(gts[hand_types[0]])):
        gt = {}
        for hand_type in hand_types:
            gt[hand_type] = gts[hand_type][i]
        old_gts.append({'joints': gt})
    return old_gts

EPE_vs = {'avg': 0}
EPE_xy_vs = {'avg': 0}
EPE_z_vs = {'avg': 0}

EPEs = {'avg': 0}

EPEs_xy = {'avg': 0}
EPEs_z = {'avg': 0}

names = get_names(folder)
for name in names:
    time_str, video_name = name.split('_')
    cur_gt_path = gt_root + time_str + '/'
    frames = load_frames(f'{gt_root}/{time_str}/{video_name}/frames/')
    gts = loadjs(f'{gt_root}/{time_str}/joints.json')
    gts = new_gt_to_old_gt(gts)

    joints = loadjs(folder + name + '/joint.json') # todo

    EPE_vs[name] = 0
    EPE_xy_vs[name] = 0
    EPE_z_vs[name] = 0

    EPEs[name] = 0
    EPEs_xy[name] = 0
    EPEs_z[name] = 0
    count_frame = 0
    count_frame_v = 0

    for frame_id in range(start_frame, len(frames)):
        hand_types = list(gts[0]['joints'].keys())
        EPE_v = 0
        EPE_xy_v = 0
        EPE_z_v = 0

        EPE = 0
        EPE_xy = 0
        EPE_z = 0
        count_v = 0
        for hand_type in hand_types:
            joint = torch.tensor(joints[frame_id]['joints'][hand_type])
            gt = torch.tensor(gts[frame_id]['joints'][hand_type])
            cur_vis, cur_EPE_v = cal_EPE_v(joint, gt, frames[frame_id])
            cur_EPE_xy_v = cal_EPE_xy_v(joint, gt, frames[frame_id])
            cur_EPE_z_v = cal_EPE_z_v(joint, gt, frames[frame_id])
            if cur_vis:
                count_v += 1
                EPE_v += cur_EPE_v
                EPE_xy_v += cur_EPE_xy_v
                EPE_z_v += cur_EPE_z_v
            EPE += cal_EPE(joint, gt)
            EPE_xy += cal_EPE_xy(joint, gt)
            EPE_z += cal_EPE_z(joint, gt)
        EPE /= len(hand_types)
        EPE_xy /= len(hand_types)
        EPE_z /= len(hand_types)
        if count_v > 0:
            EPE_v /= count_v
            EPE_xy_v /= count_v
            EPE_z_v /= count_v

        #####################################
        # check valid
        if EPE > MAX_EPE:
            EPE = MAX_EPE
            EPE_xy = MAX_Half
            EPE_z = MAX_Half
            EPE_v = MAX_EPE
            EPE_xy_v = MAX_Half
            EPE_z_v = MAX_Half
        #####################################
        
        if count_v > 0:
            EPE_vs[name] += EPE_v
            EPE_xy_vs[name] += EPE_xy_v
            EPE_z_vs[name] += EPE_z_v
            count_frame_v += 1
        EPEs[name] += EPE
        EPEs_xy[name] += EPE_xy
        EPEs_z[name] += EPE_z
        count_frame += 1
    EPE_vs[name] /= count_frame_v
    EPE_xy_vs[name] /= count_frame_v
    EPE_z_vs[name] /= count_frame_v

    EPEs[name] /= count_frame
    EPEs_xy[name] /= count_frame
    EPEs_z[name] /= count_frame
    EPE_vs['avg'] += EPE_vs[name]
    EPE_xy_vs['avg'] += EPE_xy_vs[name]
    EPE_z_vs['avg'] += EPE_z_vs[name]
    EPEs['avg'] += EPEs[name]
    EPEs_xy['avg'] += EPEs_xy[name]
    EPEs_z['avg'] += EPEs_z[name]

EPE_vs['avg'] /= (len(list(EPE_vs.keys())) - 1)
EPE_xy_vs['avg'] /= (len(list(EPE_xy_vs.keys())) - 1)
EPE_z_vs['avg'] /= (len(list(EPE_z_vs.keys())) - 1)
EPEs['avg'] /= (len(list(EPEs.keys())) - 1)
EPEs_xy['avg'] /= (len(list(EPEs_xy.keys())) - 1)
EPEs_z['avg'] /= (len(list(EPEs_z.keys())) - 1)

with open( f'exp/{cfg.exp}/eval_results.json', 'w') as f:
    json.dump({
        'EPEs': EPEs['avg'],
        'EPEs_xy': EPEs_xy['avg'],
        'EPEs_z': EPEs_z['avg'],
        'EPE_vs': EPE_vs['avg'],
        'EPE_xy_vs': EPE_xy_vs['avg'],
        'EPE_z_vs': EPE_z_vs['avg']
    }, f, indent=4)

