import time

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


# from skimage.util import random_noise


def normalize_color(rgba):
    # scale to 0, 1
    nml_rgb = rgba[..., :3].to(torch.float32) / 255
    return nml_rgb


def seg2rgb(seg: np.ndarray, cmap_name='rainbow'):
    assert isinstance(seg, np.ndarray) and np.ndim(seg) == 2, "segmentation image must be 2 dimensional ndarray"
    cmap = plt.get_cmap(cmap_name)
    rgb_img = cmap(seg / np.max(seg))[..., :3].reshape((*seg.shape, 3))
    rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)
    return rgb_img


def mask_image(image: torch.Tensor,
               patch_size: int = 8,
               missing_rate: float = 0.1,
               output_mask=False):
    """
    Args:
        image: [B, H, W, C] - Batch size, Height, Width, Channels
        patch_size: size of patches.
        missing_rate: percentage of patches missing on image. Choose floor number.
        output_mask: whether return image mask itself

    Returns:
        masked image
    """

    assert image.shape[1] % patch_size == 0 and image.shape[2] % patch_size == 0, \
        "patch_size must divisible to image size"

    h_patches = image.shape[1] // patch_size
    w_patches = image.shape[2] // patch_size
    resizer = transforms.Resize((image.shape[1], image.shape[2]), transforms.InterpolationMode.NEAREST)
    patched_mask = ~(torch.rand(image.shape[0], h_patches, w_patches, device=image.device) <= missing_rate)
    image_mask = resizer(patched_mask)[..., None]
    masked_img = image * image_mask
    if output_mask:
        return masked_img, image_mask
    return masked_img


def apply_seg_mask(image, seg_map, seg_ids, device):
    assert image.dim() == 4 and 1 <= image.shape[3] <= 3, \
        "image format: (Batch, Width, Height, Channels)"
    seg_mask = torch.zeros_like(seg_map, dtype=torch.bool, device=device)
    for seg_id in seg_ids:
        seg_mask = torch.bitwise_or(seg_mask, (seg_map == seg_id))
    seg_mask = seg_mask.to(torch.float).view(*image.shape[:-1], 1)
    masked_image = image * seg_mask
    return masked_image


def gaussian_noise(image: torch.Tensor, mean=0, var=0.1):
    noise = torch.normal(mean, var ** 0.5, image.shape, device=image.device)
    out = torch.clip(image + noise, image.min(), image.max())
    return out


def noise_mask(image: torch.Tensor,
               patch_size: int = 8,
               missing_rate: float = 0.1,
               output_mask=False):
    assert image.shape[1] % patch_size == 0 and image.shape[2] % patch_size == 0, \
        "patch_size must divisible to image size"

    h_patches = image.shape[1] // patch_size
    w_patches = image.shape[2] // patch_size
    resizer = transforms.Resize((image.shape[1], image.shape[2]), transforms.InterpolationMode.NEAREST)
    patch_mask = ~(torch.rand(image.shape[0], h_patches, w_patches, device=image.device) <= missing_rate)
    image_mask = resizer(patch_mask)[..., None]
    if patch_size == 1:
        fill_mask = ~image_mask * torch.clip(torch.rand(image.shape, device=image.device), 0, 1)
    else:
        patch_shape = (patch_mask.shape[1], patch_mask.shape[2], image.shape[3])
        fill_mask = ~patch_mask[..., None] * torch.clip(torch.rand(patch_shape, device=image.device), 0, 1)
        fill_mask = resizer(fill_mask.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    masked_img = torch.clip(image * image_mask + fill_mask, 0., 1.)
    if output_mask:
        return masked_img, image_mask
    return masked_img


if __name__ == '__main__':
    import time

    seed = 42
    m_rate = 0.2
    random_var = 0.3
    p_size = 4

    np.random.seed(seed)
    torch.manual_seed(seed)
    # p = np.random.uniform([0, 0, 0], [1, 1, 1])
    # a_img = torch.zeros(100, 128, 128, 3)
    # a_img[..., 0] = p[0]
    # a_img[..., 1] = p[1]
    # a_img[..., 2] = p[2]

    a_img = torch.tensor(plt.imread('./docs/push_box.png')[..., :3]).view(1, 128, 128, 3)

    s = time.time()
    a_masked_img = [noise_mask(a_img, patch_size=p_size, missing_rate=m_rate) for _ in range(100)][
        -1]
    print(time.time() - s)

    a_gaussian_noised_img = gaussian_noise(a_img, var=random_var)
    # random_noise(a_img, mode='gaussian', mean=0, var=random_var, clip=True, seed=seed)
    a_mixed_img = noise_mask(a_gaussian_noised_img, patch_size=p_size, missing_rate=m_rate)

    img_dict = {
        'original': a_img,
        'masked': a_masked_img,
        'gaussian noised': a_gaussian_noised_img,
        'mixed': a_mixed_img
    }

    # img_dict = {
    #     'a_masked_img0': a_masked_img[0],
    #     'a_masked_img1': a_masked_img[1],
    #     'a_masked_img2': a_masked_img[2],
    #     'a_masked_img3': a_masked_img[3]
    # }

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    for i, (name, img) in enumerate(img_dict.items()):
        fig.add_subplot(gs[i // 2, i % 2])
        plt.title(name)
        plt.imshow(img[0].reshape(128, 128, 3))
    plt.tight_layout()
    plt.show()
