import numpy as np
import torch

PI = 3.14159265358979323846
EPS = 1e-6
so3_basis = np.array([
    [[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
    [[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]],
    [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]
])

##############################
# SO3 (R), so3 (r) functions #
##############################

def invSO3(R):  # SO(3) -> SO(3)
    return R.swapaxes(-2, -1)

def Theta(R):  # SO(3) -> R^1
    return np.arccos(np.clip((np.diagonal(R, axis1=-2, axis2=-1).sum(-1) - 1) / 2, -1, 1))

def hat(w):  # R^3 -> so(3)
    return np.einsum('...i, ijk -> ...jk', w, so3_basis)

def vee(r):  # so(3) -> R^3
    return np.stack([r[..., 2, 1], r[..., 0, 2], r[..., 1, 0]], axis=-1)

def logSO3(R):  # SO(3) -> so(3)
    theta = Theta(R)
    is_regular = (np.abs(theta) >= EPS) & (np.abs(theta) <= PI - EPS)
    is_zero = (np.abs(theta) < EPS)
    is_pi = (np.abs(theta) > PI - EPS)

    r = np.zeros_like(R)
    if is_regular.any():
        r[is_regular] = (R[is_regular] - invSO3(R[is_regular])) / \
                    (2 * np.sin(theta[is_regular, None, None])) * \
                    theta[is_regular, None, None]

    if is_zero.any():
        r[is_zero] = np.zeros_like(R[is_zero, :, :])

    if is_pi.any():
        w_pi = (R[is_pi] - np.eye(3)) / 2
        w_pi = np.sqrt(np.maximum(np.diagonal(w_pi, axis1=-2, axis2=-1) + 1, 0))
        w_1 = w_pi[..., 0]
        w_2 = w_pi[..., 1] * (np.sign(R[is_pi, 0, 1]) + (w_1 == 0))
        w_3 = w_pi[..., 2] * np.sign(4 * np.sign(R[is_pi, 0, 2]) + 2 * (w_1 == 0) * np.sign(R[is_pi, 1, 2]) + 1 * (w_1 == 0) * (w_2 == 0))
        r[is_pi] = hat(np.stack([w_1, w_2, w_3], axis=-1)) * PI
    return r

def LogSO3(R):  # SO(3) -> R^3
    return vee(logSO3(R))

def Expso3(w):  # R^3 -> SO(3)
    R = np.tile(np.eye(3, dtype=w.dtype), (w.shape[:-1] + (1, 1)))
    theta = np.linalg.norm(w, axis=-1)
    is_regular = (theta >= EPS)
    if is_regular.any():
        w_norm_hat = hat(w[is_regular] / theta[is_regular, None])
        R[is_regular] += np.sin(theta[is_regular, None, None]) * w_norm_hat + \
                        (1 - np.cos(theta[is_regular, None, None])) * \
                        np.einsum('...ij,...jk->...ik', w_norm_hat, w_norm_hat)
    return R

#####################
# SE3 (T) functions #
#####################

def invSE3(T):  # SE(3) -> SE(3)
    R, p = T[..., :3, :3], T[..., :3, 3]
    invR = invSO3(R)
    invp = -np.einsum('...ij,...j->...i', invR, p)
    invT = np.tile(np.eye(4, dtype=T.dtype), (T.shape[:-2] + (1, 1)))
    invT[..., :3, :3] = invR
    invT[..., :3, 3] = invp
    return invT

def Expse3(V):  # R^6 -> SE(3)
    w, v = V[..., :3], V[..., 3:]
    theta = np.linalg.norm(w, axis=-1)
    is_regular = (theta >= EPS)
    is_zero = (theta < EPS)

    T = np.tile(np.eye(4, dtype=V.dtype), (V.shape[:-1] + (1, 1)))
    if is_regular.any():
        w_norm_hat = hat(w[is_regular] / theta[is_regular, None])
        G = (theta[is_regular, None, None] * \
            np.tile(np.eye(3, dtype=V.dtype), (V.shape[:-1] + (1, 1)))[is_regular] + \
            (1 - np.cos(theta[is_regular, None, None])) * w_norm_hat + \
            (theta[is_regular, None, None] - np.sin(theta[is_regular, None, None])) * \
            np.einsum('...ij,...jk->...ik', w_norm_hat, w_norm_hat)) / \
            theta[is_regular, None, None]
        T[is_regular, :3, :3] = Expso3(w[is_regular])
        T[is_regular, :3, 3] = np.einsum('...ij,...j->...i', G, v[is_regular])
    if is_zero.any():
        T[is_zero, :3, 3] = v[is_zero]
    return T

##############
# Kinematics #
##############

def Adjoint(T):  # SE(3) -> R^6x6
    R, p = T[..., :3, :3], T[..., :3, 3]
    return np.concatenate([
        np.concatenate([R, np.zeros_like(R)], axis=-1),
        np.concatenate([hat(p) @ R, R], axis=-1)
    ], axis=-2)

def adjoint(V):  # R^6 -> R^6x6
    w_hat, v_hat = hat(V[..., :3]), hat(V[..., 3:])
    return np.concatenate([
        np.concatenate([w_hat, np.zeros_like(w_hat)], axis=-1),
        np.concatenate([v_hat, w_hat], axis=-1)
    ], axis=-2)

def ForwardKinematics(q, S, M_ee):  # q: [..., n], S: [n, 6], M_ee: [4, 4] -> T: [..., 4, 4]
    T = np.tile(np.eye(4, dtype=q.dtype), (*q.shape[:-1], 1, 1))
    for idx, screw in enumerate(S):
        E = Expse3(screw * q[..., idx, None])
        T = np.einsum('...ij,...jk->...ik', T, E)
    return np.einsum('...ij,...jk->...ik', T, M_ee)

##################################################
# torch functions (used inside the CompVLA model) #
##################################################

def Expso3_torch(w):  # R^3 -> SO(3)
    R = torch.eye(3, dtype=w.dtype, device=w.device).expand(w.shape[:-1] + (3, 3)).clone()
    theta = torch.norm(w, dim=-1)
    is_regular = theta >= EPS
    if is_regular.any():
        w_norm = w[is_regular] / theta[is_regular].unsqueeze(-1)
        w_hat = torch.einsum('...i, ijk -> ...jk', w_norm, torch.as_tensor(so3_basis, dtype=w.dtype, device=w.device))  # hat(w_norm)
        theta_reg = theta[is_regular].unsqueeze(-1).unsqueeze(-1)
        R[is_regular] = R[is_regular] + torch.sin(theta_reg) * w_hat + \
                        (1 - torch.cos(theta_reg)) * torch.matmul(w_hat, w_hat)
    return R
