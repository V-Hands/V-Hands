import torch
import os
import json
from SkeletonModel.iktool import IKSolver, free2pose, get_swap_map, JOINT_NAMES
import SkeletonModel.mano as manoori
import glob
from utils import load_config
cfg = load_config()

exp_root = f'exp/{cfg.exp}/'

joint_root = exp_root + '/pred_joints/'
result_root = exp_root + '/ik_results/'
joint_paths = glob.glob(joint_root + '*')

def loadjs(path):
    with open(path, 'r') as f:
        return json.load(f)

modeloris = {'Right': manoori.load(model_path='SkeletonModel/mano_v1_2/models/MANO_LEFT.pkl',
                        is_right = False,
                        num_pca_comps=45,
                        batch_size=1,
                        flat_hand_mean=True),
            'Left': manoori.load(model_path='SkeletonModel/mano_v1_2/models/MANO_RIGHT.pkl',
            is_right = True,
            num_pca_comps=45,
            batch_size=1,
            flat_hand_mean=True)}
swap_map = get_swap_map(JOINT_NAMES)

for joint_path in joint_paths:
    joint_name = joint_path.split('/')[-1].split('\\')[-1].split('.')[0]
    mano_out_path = result_root + joint_name + '/mano.json'
    obj_out_path = result_root + joint_name
    joint_out_path = result_root + joint_name + '/joint.json'

    os.makedirs(obj_out_path, exist_ok=True)
    joints = loadjs(joint_path)

    solver = IKSolver()

    paras = {'para': [], 'beta': []}
    joints_out = []
    hand_meshes = []
    N_frames = len(joints)
    for i in range(N_frames):
        cur_joints = joints[i]['joints']
        cur_jointsI = joints[i]['jointsI']

        cur_jointsI = {'Left': torch.zeros(21), 'Right': torch.zeros(21)}
        for hand_type in list(cur_joints.keys()):
            cur_joints[hand_type] = torch.tensor(cur_joints[hand_type])
            cur_jointsI[hand_type] = torch.tensor(cur_jointsI[hand_type])
        _, para, betas = solver(cur_joints, cur_jointsI)

        listpara = {}
        listbetas = {}
        for hand_type in list(cur_joints.keys()):
            listpara[hand_type] = para[hand_type].tolist()
            listbetas[hand_type] = betas[hand_type].tolist()
        paras['para'].append(listpara)
        paras['beta'].append(listbetas)
        hand_meshes = []
        joint_out = {'joints': {}}
        for hand_type in list(cur_joints.keys()):
            curbeta = torch.tensor(betas[hand_type]) 
            curpara = torch.tensor(para[hand_type])
            curmodel = modeloris[hand_type]
            free = curpara[:20]
            global_orient = curpara[20:20+3].view(1, 3)
            transl = curpara[20+3:20+3+3].view(1, 3)
            curbeta = curbeta.view(1, 10)
            pose = free2pose(free, curmodel, curbeta).view(1, 45)
            output = curmodel(betas=curbeta,
                            global_orient=global_orient,
                            hand_pose=pose,
                            transl=transl,
                            return_verts=True,
                            return_tips = True)
            j_meshes = curmodel.joint_meshes(cur_joints[hand_type])
            joint_out['joints'][hand_type] = torch.tensor(output.joints[0][swap_map]).tolist()

            h_meshes = curmodel.hand_meshes(output)
            hand_meshes.append(h_meshes[0])
        joints_out.append(joint_out)

    with open(mano_out_path, 'w') as f:
        json.dump(paras, f, indent=4)

    with open(joint_out_path, 'w') as f:
        json.dump(joints_out, f, indent=4)
