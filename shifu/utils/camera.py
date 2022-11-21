import numpy as np
import torch


def intrinsic2proj2(width, height, f_x, f_y, ppx, ppy, far, near):
    """
    Returns:
        proj_matrix
    """
    proj_matrix = np.matrix([
        [2 * f_x / width, 0., 0., 0.],
        [0., 2 * f_y / height, 0., 0.],
        [(width - 2 * ppx) / width, (height - 2 * ppy) / height,
         -(far + near) / (far - near), -1],
        [0., 0., -2.0 * far * near / (far - near), 0.],
    ])
    return proj_matrix


def intrinsic2proj(w, h, f, n):
    """
    Returns:
        proj_matrix
    """
    proj_matrix = np.matrix([
        [2 * n / w, 0., 0., 0.],
        [0., 2 * n / h, 0., 0.],
        [0., 0., -(f + n) / (f - n), -1],
        [0., 0., -2. * f * n / (f - n), 0.],
    ])
    return proj_matrix


def extrinsic2view(rot_matrix, trans_matrix):
    """
    Args:
        rot_matrix: len=6 array
        trans_matrix: len=6 array

    Returns:
        view_matrix
    """
    rot_matrix = np.reshape(rot_matrix, (3, 3))
    view_matrix = np.matrix([
        [rot_matrix[0, 0], rot_matrix[1, 0], rot_matrix[2, 0], 0.],
        [rot_matrix[0, 1], rot_matrix[1, 1], rot_matrix[2, 1], 0.],
        [rot_matrix[0, 2], rot_matrix[1, 2], rot_matrix[2, 2], 0.],
        [trans_matrix[0], trans_matrix[1], trans_matrix[2], 1.],
    ])
    return view_matrix


def get_pixel_position(world_position, view_matrix, proj_matrix, width, height):
    view_proj_matrix = view_matrix @ proj_matrix
    world_4d_position = np.hstack([world_position, [1.]])  # x, y, z, w
    clipping_pos = np.array(world_4d_position @ view_proj_matrix).flatten()
    normalized_pos = np.clip(clipping_pos / clipping_pos[3], -1., 1.)
    p_x = (normalized_pos[0] + 1.) * width / 2.
    p_y = (1. - normalized_pos[1]) * height / 2.
    pixel_pos = np.int64([p_x, p_y])
    return pixel_pos


def get_pixel_position_torch(world_position, view_matrix, proj_matrix, width, height):
    device = world_position.device
    dtype = torch.float32
    view_proj_matrix = torch.tensor(view_matrix @ proj_matrix, dtype=dtype, device=device)
    world_4d_position = torch.ones(world_position.shape[0], 4, dtype=dtype, device=device)
    world_4d_position[:, :3] = world_position
    world_4d_position[:, 3] = 1.
    clipping_pos = world_4d_position @ view_proj_matrix
    normalized_pos = torch.clip(clipping_pos / clipping_pos[:, 3:3+1], -1., 1.)
    p_x = (normalized_pos[:, 0:0+1] + 1.) * width / 2.
    p_y = (1. - normalized_pos[:, 1:1+1]) * height / 2.
    pixel_pos = torch.hstack([p_x, p_y]).to(torch.int64)
    return pixel_pos


def get_world_position(pixel_position, view_matrix, proj_matrix, width, height):
    p_x = 2. * pixel_position[0] / width - 1.
    p_y = - 2. * pixel_position[1] / height - 1.
    view_proj_matrix = np.linalg.inv(view_matrix @ proj_matrix)
    world_4d_pos = np.array([p_x, p_y, 0]) @ view_proj_matrix
    world_pos = world_4d_pos[:3]
    return world_pos


if __name__ == '__main__':
    # array([[2.60508895, 0., 0., 0.],
    #        [0., 2.60508895, 0., 0.],
    #        [0., 0., -1.06896544, -0.20689656],
    #        [0., 0., -1., 0.]])
    print(intrinsic2proj(128, 128, f=3.0, n=0.1))
