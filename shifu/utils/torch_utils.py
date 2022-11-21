import torch


def free_tensor_attrs(obj):
    for attr_str in dir(obj):
        attr = eval(f'obj.{attr_str}')
        if not attr_str.startswith('__') and isinstance(attr, torch.Tensor):
            del attr
    torch.cuda.empty_cache()


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


def inverse_kinematics(dof_pos, ee_pos, ee_quat, tar_pos, tar_quat, j_ee, device, damping=0.05):
    pos_err = tar_pos - ee_pos
    # orientation error
    cc = quat_conjugate(ee_quat)
    q_r = quat_mul(tar_quat, cc)
    orn_err = q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    # solve damped least squares
    j_eef_T = torch.transpose(j_ee, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_ee @ j_eef_T + lmbda) @ dpose).view(
        dof_pos.shape[0], dof_pos.shape[1])

    tar_dof_pos = dof_pos + u
    return tar_dof_pos

