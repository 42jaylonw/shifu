import torch


def get_symmetric_orientation(rpy, n=1.):
    p = (2 * torch.pi) / n
    odd = rpy % p
    even = -rpy % p
    sym_ori = (odd < even) * odd + (even < odd) * even
    return sym_ori


def min_max_nlz(a, a_min, a_max, to_range=(-1, 1)):
    n0 = (a - a_min) / (a_max - a_min)
    ns = n0*(to_range[1] - to_range[0]) + to_range[0]
    return ns
