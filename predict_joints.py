from dataset import get_dataloder
from model import PoseNet
import torch.nn.functional as F
import torch
from utils import *
import os
import time
import json
cfg = load_config()

def get_heatmap_base(N, C, H):
    add_v = 0
    vector = []
    for N_id in range(N):
        for C_id in range(C):
            vector.append(add_v)
            add_v += H
    return torch.tensor(vector)
global_heatmap_base = get_heatmap_base(1000, 21*2, 40+2*cfg.pad).cuda()

def get_heatmap_base2(N, C, H, W):
    add_v = 0
    vector = []
    for N_id in range(N):
        for C_id in range(C):
            vector.append(add_v)
            add_v += H * W
    return torch.tensor(vector)
global_heatmap_base2 = get_heatmap_base2(1000, 21*2, 40+2*cfg.pad, 72+2*cfg.pad).cuda()

def get_heatmap_base3(N, C, H, W):
    add_v = 0
    vector = []
    for N_id in range(N):
        for C_id in range(C):
            vector.append(add_v)
        add_v += H * W
    return torch.tensor(vector)
global_heatmap_base3 = get_heatmap_base3(1000, 21*2, 40+2*cfg.pad, 72+2*cfg.pad).cuda()

def heatmap2pixels(heatmap):
    W_maxs, W_indexs = torch.max(heatmap, dim=3)
    HWmaxs, HW_indexs = torch.max(W_maxs, dim=2)
    W_result = W_indexs.view(-1)[HW_indexs.view(-1) + global_heatmap_base.view(-1)[:HW_indexs.view(-1).shape[0]]]
    W_result = W_result.view(heatmap.shape[0], heatmap.shape[1], 1)
    H_result = HW_indexs.view(heatmap.shape[0], heatmap.shape[1], 1)
    return torch.cat([W_result, H_result], dim=2).float()

def get_heatmap_values(heatmap, frames_pad, Ws, Hs):
    N, C, HM, WM = heatmap.shape
    heatmap = F.relu(heatmap.view(-1)) + 1e-8
    frames_pad = torch.tensor(frames_pad[:,0]).view(-1)
    heatmap_indexs = Hs.view(-1) * WM + Ws.view(-1) + global_heatmap_base2.view(-1)[:N*C]
    heatmap_indexs = heatmap_indexs.long()
    frame_indexs = Hs.view(-1) * WM + Ws.view(-1) + global_heatmap_base3.view(-1)[:N*C]
    frame_indexs = frame_indexs.long()
    return heatmap[heatmap_indexs].view(N, C), frames_pad[frame_indexs].view(N, C)

def meanpixels(joint_pixels, heatmap, frames_pad):
    N, C, HM, WM = heatmap.shape
    Wc = joint_pixels[:,:,0]
    Hc = joint_pixels[:,:,1]
    W_sum = 0
    H_sum = 0
    I_sum = 0
    count = 0
    for deltaW, deltaH in zip([-1, 0, 1], [-1, 0, 1]):
        Ws = Wc + deltaW
        Hs = Hc + deltaH
        Ws[Ws >= WM] = WM - 1
        Ws[Ws < 0] = 0
        Hs[Hs >= HM] = HM - 1
        Hs[Hs < 0] = 0
        heatmap_value, I_value = get_heatmap_values(heatmap, frames_pad, Ws, Hs)
        W_sum += Ws * heatmap_value
        H_sum += Hs * heatmap_value
        I_sum += I_value * heatmap_value
        count += heatmap_value
    Wm = W_sum / count
    Hm = H_sum / count
    Im = I_sum / count
    Im[Im > 0.3] = 1
    Im[Im < 0.5] = 0
    return torch.cat([Wm.unsqueeze(2), Hm.unsqueeze(2)], dim=2).float(), Im

def heatmap2joints(heatmap, frames_pad, screen_w, screen_h, H=40, W=72):
    joint_pixels = heatmap2pixels(heatmap)
    joint_pixels, Im = meanpixels(joint_pixels, heatmap, frames_pad)
    joint_pixels = joint_pixels - cfg.pad + 0.5
    joint_pixels[:,:,0] = joint_pixels[:,:,0] * screen_w / W
    joint_pixels[:,:,1] = joint_pixels[:,:,1] * screen_h / H
    return joint_pixels.cuda(), Im

def joints2to3(joints2d, jointsz):
    joints3d = torch.zeros((joints2d.shape[0], joints2d.shape[1], 3)).cuda()
    joints3d[:,:,:2] += joints2d
    joints3d[:,:,2] += jointsz
    return joints3d

def cal_acc(type_pred, joint_type_gt):
    type_pred = type_pred.detach().clone()
    type_pred[type_pred > 0.5] = 1
    type_pred[type_pred < 0.6] = 0
    same_map = type_pred * joint_type_gt + (1 - type_pred) * (1 - joint_type_gt)
    return torch.sum(same_map) / joint_type_gt.shape[0] / 2

def swap_LR(tensor):
    tensor = tensor.clone().detach()
    tensor_old_left = tensor[:,:21].clone().detach()
    tensor[:,:21] = tensor[:,21:]
    tensor[:,21:] = tensor_old_left
    return tensor

def flippred(model, frames_pad):
    heatmap_preds, joints_z_preds, existence_preds = [], [], []
    hs = None
    hsf = None
    for t in range(0, frames_pad.shape[1]):
        cur_frames_pad = frames_pad[:,t]
        heatmap_pred, joints_z_pred, existence_pred, hs = model(cur_frames_pad, hs)
        cur_frames_padf = cur_frames_pad.flip(3)
        heatmap_predf, joints_z_predf, existence_predf, hsf = model(cur_frames_padf, hsf)
        heatmap_predf = heatmap_predf.flip(3)
        heatmap_predf, joints_z_predf, existence_predf = swap_LR(heatmap_predf), swap_LR(joints_z_predf), existence_predf.flip(1)
        heatmap_preds.append((heatmap_pred + heatmap_predf).unsqueeze(1) / 2)
        joints_z_preds.append((joints_z_pred + joints_z_predf).unsqueeze(1) / 2)
        existence_preds.append((existence_pred + existence_predf).unsqueeze(1) / 2)
    return torch.cat(heatmap_preds, dim=1), torch.cat(joints_z_preds, dim=1), torch.cat(existence_preds, dim=1)

def memory_efficient_pred(model, frames_pad):
    heatmap_preds, joints_z_preds, existence_preds = [], [], []
    N = frames_pad.shape[0]
    for i in range(N):
        heatmap_pred, joints_z_pred, existence_pred = flippred(model, frames_pad[i:i+1])
        heatmap_preds.append(heatmap_pred), joints_z_preds.append(joints_z_pred), existence_preds.append(existence_pred)
    return torch.cat(heatmap_preds, dim=0), torch.cat(joints_z_preds, dim=0), torch.cat(existence_preds, dim=0)

def test(test_loader, model):
    mse_loss_sum = 0
    acc_sum = 0
    count = 0
    video_num = 0
    os.makedirs('exp/' + cfg.exp + '/pred_joints/', exist_ok=True)
    test_results = {}
    for video_id, (frames, joints_gt, existences, screen_w, screen_h, frames_pad, folder_name, video_name) in enumerate(test_loader):
        screen_w, screen_h = float(screen_w), float(screen_h)
        existences = existences.view(-1, 2).cuda()
        joints_gt = joints_gt.view(-1, 21*2, 3).cuda()
        frames = frames.view(-1, 1, frames.shape[3], frames.shape[4]).cuda()
        count += frames_pad.shape[1]
        video_num += 1
        with torch.no_grad():
            heatmap_pred, joints_z_pred, existence_pred = memory_efficient_pred(model, frames_pad)
            heatmap_pred = heatmap_pred.view(-1, 21*2, frames_pad.shape[3], frames_pad.shape[4])
            joints_z_pred = joints_z_pred.view(-1, 21*2, 1)
            existence_pred = existence_pred.view(-1, 2)
            joints2d_pred, jointsI = heatmap2joints(heatmap_pred.detach().view(-1, 21*2, frames_pad.shape[3], frames_pad.shape[4]), frames_pad.view(-1, 2, frames_pad.shape[3], frames_pad.shape[4]), screen_w, screen_h)
            joints_pred = joints2to3(joints2d_pred.view(-1, 21*2, 2), joints_z_pred.view(-1, 21*2)).view(-1, 21*2, 3)
            joints_dif = joints_pred - joints_gt
            joints_dif[:, :21, :] *= existences[:, 0].view(-1, 1, 1)
            joints_dif[:, 21:, :] *= existences[:, 1].view(-1, 1, 1)
            existences_rate = torch.mean(existences)
            joints_loss = torch.mean(joints_dif * joints_dif) / existences_rate
            acc = cal_acc(existence_pred, existences)
            mse_loss_sum += joints_loss * frames_pad.shape[1]
            acc_sum += acc * frames_pad.shape[1]
        test_results[folder_name[0] + video_name[0]] = {
            'mse': float(joints_loss.cpu()),
            'acc': float(acc.cpu()),
        }
        joints_output = []
        for frame_id in range(frames.shape[0]):
            cur_joints_pred = {'joints': {}, 'jointsI': {}}
            if existences[frame_id][0] > 0.5:
                cur_joints_pred['joints']['Left'] = joints_pred[frame_id][:21].cpu().tolist()
                cur_joints_pred['jointsI']['Left'] = jointsI[frame_id][:21].cpu().tolist()
            if existences[frame_id][1] > 0.5:
                cur_joints_pred['joints']['Right'] = joints_pred[frame_id][21:].cpu().tolist()
                cur_joints_pred['jointsI']['Right'] = jointsI[frame_id][21:].cpu().tolist()
            joints_output.append(cur_joints_pred)
        with open('exp/' + cfg.exp + '/pred_joints/{}.json'.format(folder_name[0] + '_' + video_name[0]), 'w') as f:
            json.dump(joints_output, f, indent=4)
    mse_loss = tofloat(mse_loss_sum / count)
    acc = tofloat(acc_sum / count)
    return mse_loss, acc, test_results

if __name__ == '__main__':
    test_loader = get_dataloder(cfg.test_root, 'test', 1, 1) 
    model = PoseNet().cuda()
    model.load_state_dict(torch.load('exp/' + cfg.exp + '/checkpoints/{}.pth'.format(cfg.epochs)))
    cfg.exp = cfg.exp
    model.eval()
    time_start = time.time()
    mse_loss, acc, test_results = test(test_loader, model)
    print('epoch: {}, test mse: {}, test acc: {}'.format(cfg.epochs, mse_loss, acc))
    time_end = time.time()

    result = {
        'result_each_clip': test_results,
        'mse_loss': mse_loss,
        'acc': acc
    }

    with open('exp/' + cfg.exp + '/test_result.json', 'w') as f:
        json.dump(result, f, indent=2)

