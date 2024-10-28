from dataset import get_dataloder
from model import PoseNet
import torch.nn.functional as F
import torch
from utils import *
import os
import time
import torch.optim.lr_scheduler as lr_scheduler
import random
cfg = load_config()

def get_heatmap_base(N, C, H):
    add_v = 0
    vector = []
    for N_id in range(N):
        for C_id in range(C):
            vector.append(add_v)
            add_v += H
    return torch.tensor(vector)
global_heatmap_base = get_heatmap_base(cfg.batch*cfg.frame_len, 21*2, 40+2*cfg.pad).cuda()

def heatmap2pixels(heatmap):
    W_maxs, W_indexs = torch.max(heatmap, dim=3)
    HWmaxs, HW_indexs = torch.max(W_maxs, dim=2)
    W_result = W_indexs.view(-1)[HW_indexs.view(-1) + global_heatmap_base.view(-1)[:HW_indexs.view(-1).shape[0]]]
    W_result = W_result.view(heatmap.shape[0], heatmap.shape[1], 1)
    H_result = HW_indexs.view(heatmap.shape[0], heatmap.shape[1], 1)
    return torch.cat([W_result, H_result], dim=2).float()

def heatmap2joints(heatmap, screen_w, screen_h, H=40, W=72):
    joint_pixels = heatmap2pixels(heatmap)
    joint_pixels = joint_pixels - cfg.pad + 0.5
    joint_pixels[:,:,0] *= screen_w.cuda() / W
    joint_pixels[:,:,1] *= screen_h.cuda() / H
    return joint_pixels.cuda()

def joints2to3(joints2d, jointsz):
    joints3d = torch.zeros((joints2d.shape[0], joints2d.shape[1], 3)).cuda()
    joints3d[:,:,:2] += joints2d
    joints3d[:,:,2] += jointsz
    return joints3d

def joints2bones_(joints):
    joints0 = joints[:, :, 0:1]
    joints1to4 = joints[:, :, 1:].view(joints.shape[0], joints.shape[1], 5, 4, 3)
    bones0 = joints1to4[:, :, :, 0] - joints0
    bones123 = (joints1to4[:, :, :, 1:] - joints1to4[:, :, :, :-1]).view(joints.shape[0], joints.shape[1], -1, 3)
    bonesmid1 = joints[:, :, 5:6] - joints[:, :, 9:10]
    bonesmid2 = joints[:, :, 9:10] - joints[:, :, 13:14]
    bonesmid3 = joints[:, :, 13:14] - joints[:, :, 17:18]
    bones = torch.cat([bones0, bones123, bonesmid1, bonesmid2, bonesmid3], dim=2)
    bones = torch.sqrt(torch.sum(bones * bones, dim=3) + 1e-8)
    return bones

def joints2bones(joints):
    return torch.cat([joints2bones_(joints[:, :, :21]), joints2bones_(joints[:, :, 21:])], dim=2)

def weighted_loss(heatmap_pred, heatmaps, joints_z_pred, joints_z_gt, bones_gt, existence_pred, existences, screen_w, screen_h, heatmap_weight=10., depth_weight=2., bone_weight=1., fweight=1.):
    N, T = joints_z_gt.shape[:2]

    heatmap_dif = heatmap_pred - heatmaps
    joints_z_dif = joints_z_pred - joints_z_gt
    
    screen_w = screen_w.view(N, 1).repeat(1, T).view(-1, 1)
    screen_h = screen_h.view(N, 1).repeat(1, T).view(-1, 1)

    joints2d_pred = heatmap2joints(heatmap_pred.detach().view(-1, 21*2, heatmap_pred.shape[3], heatmap_pred.shape[4]), screen_w, screen_h)
    joints_pred = joints2to3(joints2d_pred.view(-1, 21*2, 2), joints_z_pred.view(-1, 21*2)).view(N, T, 21*2, 3)
    bones = joints2bones(joints_pred)
    bone_dif = bones - bones_gt

    heatmap_dif[:, :, :21, :, :] *= existences[:, :, 0].view(existences.shape[0], existences.shape[1], 1, 1, 1)
    heatmap_dif[:, :, 21:, :, :] *= existences[:, :, 1].view(existences.shape[0], existences.shape[1], 1, 1, 1)
    joints_z_dif[:, :, :21] *= existences[:, :, 0].view(existences.shape[0], existences.shape[1], 1)
    joints_z_dif[:, :, 21:] *= existences[:, :, 1].view(existences.shape[0], existences.shape[1], 1)
    bone_dif[:, :, :23] *= existences[:, :, 0].view(existences.shape[0], existences.shape[1], 1)
    bone_dif[:, :, 23:] *= existences[:, :, 1].view(existences.shape[0], existences.shape[1], 1)

    existences_rate = torch.mean(existences)
    heatmap_loss = torch.mean(heatmap_dif * heatmap_dif) / existences_rate
    joints_z_loss = torch.mean(joints_z_dif * joints_z_dif) / existences_rate
    bone_loss = torch.mean(bone_dif * bone_dif) / existences_rate
    mse_loss = heatmap_weight * heatmap_loss + depth_weight * joints_z_loss + bone_weight * bone_loss
    binary_cross_entropy = F.binary_cross_entropy(existence_pred, existences)

    existences_rate = torch.mean(existences[:,:2])
    heatmap_loss = torch.mean(heatmap_dif[:,:2] * heatmap_dif[:,:2]) / existences_rate
    joints_z_loss = torch.mean(joints_z_dif[:,:2] * joints_z_dif[:,:2]) / existences_rate
    bone_loss = torch.mean(bone_dif[:,:2] * bone_dif[:,:2]) / existences_rate
    mse_lossf = heatmap_weight * heatmap_loss + depth_weight * joints_z_loss + bone_weight * bone_loss
    binary_cross_entropyf = F.binary_cross_entropy(existence_pred[:,:2], existences[:,:2])

    return mse_loss + fweight * mse_lossf, binary_cross_entropy + fweight * binary_cross_entropyf

def swap_LR(tensor):
    tensor = tensor.clone().detach()
    tensor_old_left = tensor[:,:,:21].clone().detach()
    tensor[:,:,:21] = tensor[:,:,21:]
    tensor[:,:,21:] = tensor_old_left
    return tensor

def swap_LRB(tensor):
    tensor = tensor.clone().detach()
    tensor_old_left = tensor[:,:,:23].clone().detach()
    tensor[:,:,:23] = tensor[:,:,23:]
    tensor[:,:,23:] = tensor_old_left
    return tensor

def data_aug(frames_pad, heatmaps, joints_z_gt, bones_gt, existences, method):
    if method == 'None':
        return frames_pad, heatmaps, joints_z_gt, bones_gt, existences
    elif method=='LR':
        frames_pad = frames_pad.flip(4)
        heatmaps = heatmaps.flip(4)
        heatmaps = swap_LR(heatmaps)
        joints_z_gt = swap_LR(joints_z_gt)
        bones_gt = swap_LRB(bones_gt)
        existences = existences.flip(2)
    else:
        raise NotImplementedError
    
    return frames_pad, heatmaps, joints_z_gt, bones_gt, existences

def pred_results(model, frames_pad):
    heatmap_preds, depth_preds, existence_preds = [], [], []
    hs = None
    for t in range(0, frames_pad.shape[1]):
        cur_frames_pad = frames_pad[:,t]
        heatmap_pred, depth_pred, existence_pred, hs = model(cur_frames_pad, hs)
        heatmap_preds.append(heatmap_pred.unsqueeze(1))
        depth_preds.append(depth_pred.unsqueeze(1))
        existence_preds.append(existence_pred.unsqueeze(1))
    
    heatmap_preds = torch.cat(heatmap_preds, dim=1)
    return heatmap_preds, torch.cat(depth_preds, dim=1), torch.cat(existence_preds, dim=1)

def get_loss(model, cur_frames_pad, cur_heatmaps, cur_joints_z_gt, cur_bones_gt, cur_existences, screen_w, screen_h):
    heatmap_pred, joints_z_pred, existence_pred = pred_results(model, cur_frames_pad)
    mse_loss, binary_cross_entropy = weighted_loss(heatmap_pred, cur_heatmaps, joints_z_pred, cur_joints_z_gt, cur_bones_gt, existence_pred, cur_existences, screen_w, screen_h)
    return mse_loss, binary_cross_entropy

def train_one(train_loader, model, optimizer):
    mse_loss_sum = 0
    binary_cross_entropy_sum = 0
    count = 0
    for frames, joints_gt, existences, screen_w, screen_h, heatmaps, frames_pad in train_loader:
        joints_z_gt = joints_gt[:,:,:,2]
        bones_gt = joints2bones(joints_gt)
        optimizer.zero_grad()
        aug_methods = ['None', 'LR']

        cur_frames_pad, cur_heatmaps, cur_joints_z_gt, cur_bones_gt, cur_existences = data_aug(frames_pad, heatmaps, joints_z_gt, bones_gt, existences, random.choice(aug_methods))

        model.train()
        mse_loss, binary_cross_entropy = get_loss(model, cur_frames_pad, cur_heatmaps, cur_joints_z_gt, cur_bones_gt, cur_existences, screen_w, screen_h)
        loss = mse_loss + 0.2 * binary_cross_entropy

        loss.backward()
        mse_loss_sum += float(mse_loss.cpu() * frames_pad.shape[0])
        binary_cross_entropy_sum += float(binary_cross_entropy.cpu() * frames_pad.shape[0])
        count += float(frames_pad.shape[0])
        
        optimizer.step()
    return mse_loss_sum / count, binary_cross_entropy_sum / count

if __name__ == '__main__':
    os.makedirs('exp/' + cfg.exp, exist_ok=True)

    train_loader = get_dataloder(cfg.train_root, 'train', cfg.frame_len, cfg.batch)
    train_loader.dataset.frame_len = 2

    train_vis_epochs = 10

    model = PoseNet().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.999)

    train_results = []
    time_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        frame_len = 2 + (epoch - 1) // 20
        if frame_len <= cfg.frame_len and frame_len > train_loader.dataset.frame_len:
            train_loader.dataset.frame_len = frame_len
            global_heatmap_base = get_heatmap_base(cfg.batch*frame_len, 21*2, 40+2*cfg.pad).cuda()
            print('train_loader.dataset.frame_len', train_loader.dataset.frame_len)
        mse_loss, binary_cross_entropy = train_one(train_loader, model, optimizer)
        train_results.append([mse_loss, binary_cross_entropy])
        print('epoch {}, train mse_loss {}, binary_cross_entropy {}'.format(epoch, mse_loss, binary_cross_entropy))
        if epoch % train_vis_epochs == 0:
            with open('exp/' + cfg.exp + '/train_results.txt', 'w') as f:
                f.write(
                    '\n'.join([
                        str(epoch_id + 1) + ' ' + \
                        '\t'.join([
                            str(cur_result) 
                        for cur_result in results]) 
                    for epoch_id, results in enumerate(train_results)])
                )
        scheduler.step()

    os.makedirs('exp/' + cfg.exp + '/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'exp/' + cfg.exp + '/checkpoints/{}.pth'.format(epoch))
