import torch
from SkeletonModel import mano_fast
import torch.nn.functional as F

JOINT_NAMES = {
    'wrist': 0,
    'index1': 1,
    'index2': 2,
    'index3': 3,
    'middle1': 4,
    'middle2': 5,
    'middle3': 6,
    'pinky1': 7,
    'pinky2': 8,
    'pinky3': 9,
    'ring1': 10,
    'ring2': 11,
    'ring3': 12,
    'thumb1': 13,
    'thumb2': 14,
    'thumb3': 15,    
    'thumb_tip': 16,
    'index_tip': 17,
    'middle_tip': 18,
    'ring_tip': 19,
    'pinky_tip': 20
}

def get_swap_map(JOINT_NAMES):
    swap_map = [
        JOINT_NAMES['wrist'],

        JOINT_NAMES['thumb1'],
        JOINT_NAMES['thumb2'],
        JOINT_NAMES['thumb3'],
        JOINT_NAMES['thumb_tip'],

        JOINT_NAMES['index1'],
        JOINT_NAMES['index2'],
        JOINT_NAMES['index3'],
        JOINT_NAMES['index_tip'],

        JOINT_NAMES['middle1'],
        JOINT_NAMES['middle2'],
        JOINT_NAMES['middle3'],
        JOINT_NAMES['middle_tip'],

        JOINT_NAMES['ring1'],
        JOINT_NAMES['ring2'],
        JOINT_NAMES['ring3'],
        JOINT_NAMES['ring_tip'],

        JOINT_NAMES['pinky1'],
        JOINT_NAMES['pinky2'],
        JOINT_NAMES['pinky3'],
        JOINT_NAMES['pinky_tip'],
    ]
    return swap_map

def stdvec(vec):
    return vec / vec.norm()

def get_matrix(joints):
    matrix = torch.zeros((45, 20), dtype=torch.float32)
    iy = torch.tensor([0,-1,0], dtype=torch.float32)
    for i in range(5):
        cur = i * 3 + 1
        r = stdvec(joints[cur + 1] - joints[cur])
        matrix[(cur-1)*3:cur*3, i*2] += iy
        matrix[(cur-1)*3:cur*3, i*2 + 1] += torch.cross(r, iy)
    for i in range(5):
        cur = i * 3 + 2
        r = stdvec(joints[cur + 1] - joints[cur])
        if i < 4:
            matrix[(cur-1)*3:cur*3, i + 10] += torch.cross(r, iy)
        else:
            matrix[(cur-1)*3:cur*3, i + 10] += stdvec(torch.cross(r, iy) + iy)
    for i in range(5):
        cur = i * 3 + 3
        curc = [17, 18, 20, 19, 16][i]
        r = stdvec(joints[curc] - joints[cur])
        if i < 4:
            matrix[(cur-1)*3:cur*3, i + 15] += torch.cross(r, iy)
        else:
            matrix[(cur-1)*3:cur*3, i + 15] += stdvec(torch.cross(r, iy) + iy)
    return matrix

def free2pose(free, model, betas):
    iy = torch.tensor([0,-1,0], dtype=torch.float32)
    with torch.no_grad():
        joints = model(betas=betas,
                        global_orient=torch.zeros((1, 3)),
                        hand_pose=torch.zeros((1, 45)),
                        transl=torch.zeros((1, 3)),
                        return_verts=True,
                        return_tips = True).joints[0]
    pose = torch.zeros((45))
    for i in range(5):
        cur = i * 3 + 1
        r = stdvec(joints[cur + 1] - joints[cur])
        pose[(cur-1)*3:cur*3] += iy * free[i*2]
        pose[(cur-1)*3:cur*3] += torch.cross(r, iy) * free[i*2 + 1]
    for i in range(5):
        cur = i * 3 + 2
        r = stdvec(joints[cur + 1] - joints[cur])
        if i < 4:
            pose[(cur-1)*3:cur*3] += torch.cross(r, iy) * free[i + 10]
        else:
            pose[(cur-1)*3:cur*3] += stdvec(torch.cross(r, iy) + iy) * free[i + 10]
    for i in range(5):
        cur = i * 3 + 3
        curc = [17, 18, 20, 19, 16][i]
        r = stdvec(joints[curc] - joints[cur])
        if i < 4:
            pose[(cur-1)*3:cur*3] += torch.cross(r, iy) * free[i + 15]
        else:
            pose[(cur-1)*3:cur*3] += stdvec(torch.cross(r, iy) + iy) * free[i + 15]
    return pose

min_vs = -torch.tensor([0.01,0.01,0.01,0.01,0.01,0.15,0.05,0.15,0.05,1.05], dtype=torch.float32)
def regular_para(mano_para):
    bound = mano_para[10:20] < min_vs
    mano_para[10:20][bound] = min_vs[bound]
    return mano_para, bound

def regular_joint(mano_para, joints, jacobian, z_M=-0.004, sigma=1e-3):
    boundj = joints > z_M
    boundj = boundj.view(21, 3)
    boundj[:, :2] = False
    boundj = boundj.view(-1)
    B = jacobian[boundj, :]
    t = (z_M - joints)[boundj]
    return mano_para + B.T @ torch.linalg.inv(B @ B.T + sigma * torch.eye(B.shape[0])) @ t, boundj

def solve_para(jointsGT, model, matrix, J, v_shaped, swap_map, mano_para=None, sigma=1e-3, min_loss=2e-5, weight=None, bound=None, boundj=None, z_M=-0.004):
    for iter in range(2):
        def para2joint(mano_para):
            free = mano_para[:20]
            global_orient = mano_para[20:20+3].view(1, -1)
            transl = mano_para[20+3:].view(1, -1)
            pose = (matrix @ free).view(1, 45)
            joints = model(global_orient=global_orient,
                            hand_pose=pose,
                            transl=transl,
                            return_verts=True,
                            return_tips = True,
                            J=J, 
                            v_shaped=v_shaped).joints
            joints = joints[0][swap_map].view(-1)                   
            return joints.view(-1)
        joints = para2joint(mano_para)
        delta = jointsGT - joints

        with torch.no_grad():
            jacobian = torch.autograd.functional.jacobian(para2joint, mano_para, create_graph=False, vectorize=True)

        if not bound == None:
            vpara = jacobian[:, 10:20].T @ delta
            bound = bound & (vpara < -1e-8)
            jacobian[:, 10:20][:, bound] = 0
        if not weight == None:
            delta *= weight
            jacobian *= weight.view(-1, 1)
        if not boundj == None:
            vjoint = jacobian[boundj] @ jacobian.T @ delta
            boundj[boundj.clone()] = boundj[boundj] & (vjoint > 1e-8)
            B = jacobian[boundj, :]
            E = torch.linalg.inv(jacobian.T @ jacobian + sigma * torch.eye(26))
            H = torch.linalg.inv(B @ E @ B.T + sigma * torch.eye(B.shape[0]))
            q = jacobian.T @ delta
            t = (z_M - joints)[boundj]
            mano_para = mano_para + E @ (q - B.T @ H @ (B @ E @ q - t))
        else:
            mano_para = mano_para + torch.linalg.inv(jacobian.T @ jacobian + sigma * torch.eye(26)) @ jacobian.T @ delta

        mano_para, bound_new = regular_para(mano_para)
        bound = bound | bound_new
        jacobian[:, 10:20][:, bound] = 0
        joints = para2joint(mano_para)
        mano_para, boundj_new = regular_joint(mano_para, joints, jacobian, z_M=z_M)
        if boundj == None:
            boundj = boundj_new
        else:
            boundj = boundj | boundj_new
        while True in boundj_new.tolist():
            joints = para2joint(mano_para)
            mano_para, bound_new = regular_para(mano_para)
            bound = bound | bound_new
            if True in bound_new.tolist():
                jacobian[:, 10:20][:, bound] = 0
                mano_para, boundj_new = regular_joint(mano_para, joints, jacobian, z_M=z_M)
                boundj = boundj | boundj_new
            else:
                break
        loss = F.mse_loss(joints, jointsGT)
        if loss < min_loss:
            break

    return joints.view(21, 3), mano_para, bound, boundj
    
def init_betas(jointsGT, model, swap_map, sigma=1e-3, min_loss=1e-5, beta_max=1., bound=None):
    betas_mano_para = torch.zeros((20+3+3+10))
    for iter in range(20):
        def para2joint(betas_mano_para):
            free = betas_mano_para[:20]
            global_orient = betas_mano_para[20:20+3].view(1, -1)
            transl = betas_mano_para[20+3:20+3+3].view(1, -1)
            betas = betas_mano_para[20+3+3:].view(1, -1)
            pose = free2pose(free, model, betas).view(1, 45)
            joints = model(betas=betas,
                            global_orient=global_orient,
                            hand_pose=pose,
                            transl=transl,
                            return_verts=True,
                            return_tips = True).joints
            joints = joints[0][swap_map].view(-1)                   
            return joints.view(-1)
        joints = para2joint(betas_mano_para)
        delta = jointsGT - joints
        loss = F.mse_loss(joints, jointsGT)
        if loss < min_loss:
            break
        jacobian = torch.autograd.functional.jacobian(para2joint, betas_mano_para, create_graph=False, vectorize=True)
        if not bound == None:
            ngradient = jacobian[:, 10:20].T @ delta
            jacobian[:, 10:20][:, bound & (ngradient < -1e-8)] = 0
        betas_mano_para = betas_mano_para + torch.linalg.inv(jacobian.T @ jacobian + sigma * torch.eye(36)) @ jacobian.T @ delta
        betas_mano_para[20+3+3:][betas_mano_para[20+3+3:] > beta_max] = beta_max
        betas_mano_para[20+3+3:][betas_mano_para[20+3+3:] < -beta_max] = -beta_max
        betas_mano_para[:20+3+3], bound = regular_para(betas_mano_para[:20+3+3])
    return joints.view(21, 3), betas_mano_para[:20+3+3], betas_mano_para[20+3+3:], bound, None

def update_betas(jointsGT, model, mano_para, betas, swap_map, sigma=1e-3, beta_max=1.):
    def betas2joint(para_betas):
        free = mano_para[:20]
        global_orient = mano_para[20:20+3].view(1, -1)
        transl = mano_para[20+3:].view(1, -1)
        pose = free2pose(free, model, betas.view(1, 10)).view(1, 45)
        joints = model(betas=para_betas.view(1, 10),
                        global_orient=global_orient,
                        hand_pose=pose,
                        transl=transl,
                        return_verts=True,
                        return_tips = True).joints
        joints = joints[0][swap_map].view(-1)                   
        return joints.view(-1)
    joints = betas2joint(betas)
    delta = jointsGT - joints
    jacobian = torch.autograd.functional.jacobian(betas2joint, betas, create_graph=False, vectorize=True)
    betas = betas + torch.linalg.inv(jacobian.T @ jacobian + sigma * torch.eye(10)) @ jacobian.T @ delta
    betas[betas > beta_max] = beta_max
    betas[betas < -beta_max] = -beta_max
    return betas

def joints2mano(jointsGT, model, mano_para, betas, swap_map, preframes=0, bound=None, boundj=None, weight=None, z_M=-0.004, fix_beta_num=500):
    jointsGT = jointsGT.view(-1)
    if betas == None:
        return init_betas(jointsGT, model, swap_map)
    else:
        Tjoints = model(betas=betas.view(1, 10),
                        global_orient=torch.zeros((1, 3)),
                        hand_pose=torch.zeros((1, 45)),
                        transl=torch.zeros((1, 3)),
                        return_verts=True,
                        return_tips=True).joints[0]
        matrix = get_matrix(Tjoints)
        J, v_shaped = model.get_shape_var(betas.view(1, 10))
        mano_joints, mano_para, bound, boundj = solve_para(jointsGT, model, matrix, J, v_shaped, swap_map, mano_para, bound=bound, boundj=boundj, weight=weight, z_M=z_M)
        if preframes < fix_beta_num:
            cur_betas = update_betas(jointsGT, model, mano_para, betas, swap_map, sigma=1e-3)
            betas = (betas * preframes + cur_betas) / (preframes + 1)
        return mano_joints, mano_para, betas, bound, boundj

def get_on_screen(jointsGT, jointsI, ids_L1=torch.tensor([5,9,13,17]), ids_L4=torch.tensor([4,8,12,16,20]), z_M=0.005, z_s=-0.045):
    depth = jointsGT[:, 2].detach().clone()
    depth[jointsI < 0.5] = -1000
    on_screen = torch.zeros((21))
    on_screen[ids_L4[depth[ids_L4] > z_s]] = 1
    jointsGT[:,2][on_screen > 0.5] = z_M
    if torch.sum(jointsI[ids_L1]) >= 3:
        if torch.mean(jointsGT[:,2][ids_L1]) > -0.025:
            on_screen[ids_L1[jointsI[ids_L1] > 0.5]] = 1
            jointsGT[ids_L1[on_screen[ids_L1] > 0.5], 2] += 0.013
    jointsGT[:,2][jointsGT[:,2] > z_M] = z_M
    return on_screen, jointsGT

class IKSolver:
    def init_vars(self):
        self.mano_para = {'Left': None, 'Right': None}
        self.betas = {'Left': None, 'Right': None}
        self.bound = {'Left': None, 'Right': None}
        self.boundj = {'Left': None, 'Right': None}
        self.preframes = 0

    def __init__(self):
        self.swap_map = get_swap_map(JOINT_NAMES)
        self.models = {'Right': mano_fast.load(model_path='SkeletonModel/mano_v1_2/models/MANO_LEFT.pkl',
                        is_right = False,
                        num_pca_comps=45,
                        batch_size=1,
                        flat_hand_mean=True),
                        'Left': mano_fast.load(model_path='SkeletonModel/mano_v1_2/models/MANO_RIGHT.pkl',
                        is_right = True,
                        num_pca_comps=45,
                        batch_size=1,
                        flat_hand_mean=True)}
        self.init_vars()
        self.finish_init = False
        self.z_M = -0.001

    def __call__(self, jointsGT, jointsI):
        if (not self.finish_init) and (self.preframes == 10):
            self.init_vars()
            self.finish_init = True

        mano_joints = {}
        for hand_type in list(jointsGT.keys()):
            cur_jointsGT = jointsGT[hand_type].view(-1, 3)
            cur_jointsI = jointsI[hand_type].view(-1)
            on_screen, cur_jointsGT = get_on_screen(cur_jointsGT, cur_jointsI, z_M=self.z_M)
            weight = torch.ones((cur_jointsGT.shape[0]), dtype=torch.float32)
            weight[on_screen > 0.5] = 10.
            weight = weight.view(-1, 1).repeat(1, 3).view(-1)
            mano_joints[hand_type], self.mano_para[hand_type], self.betas[hand_type], self.bound[hand_type], self.boundj[hand_type] = \
                joints2mano(cur_jointsGT, self.models[hand_type], self.mano_para[hand_type], self.betas[hand_type], self.swap_map, preframes=self.preframes, bound=self.bound[hand_type], boundj=self.boundj[hand_type], weight=weight, z_M=self.z_M)
        self.preframes += 1
        return mano_joints, self.mano_para, self.betas
