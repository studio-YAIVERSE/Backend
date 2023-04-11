
from .nn import get_generator_ema, get_device


import torch
import numpy as np
import os
import PIL.Image
from training.utils.utils_3d import save_obj, savemeshtes2
import cv2


@torch.inference_mode()
def inference(
        run_dir='.',  # Output directory.
):
    # from rest_framework.exceptions import PermissionDenied
    # if not settings.TORCH_ENABLED:
    #     raise PermissionDenied("Server started without pytorch initialization.")

    device = get_device()
    G_ema = get_generator_ema()

    geo_z = torch.randn([1, G_ema.z_dim], device=device)
    tex_z = torch.randn([1, G_ema.z_dim], device=device)

    G_ema.update_w_avg(None)
    G_ema.generate_3d_mesh(
        geo_z=geo_z, tex_z=tex_z, c=None, truncation_psi=0.7,
        use_style_mixing=False
    )

    print('==> generate inference 3d shapes with texture')

    c_to_compute_w_avg = None
    G_ema.update_w_avg(c_to_compute_w_avg)
    save_mesh_idx = 0
    mesh_dir = run_dir
    generated_mesh = G_ema.generate_3d_mesh(
        geo_z=geo_z, tex_z=tex_z, c=None, truncation_psi=0.7,
        use_style_mixing=False
    )
    (mesh_v,), (mesh_f,), (all_uvs,), (all_mesh_tex_idx,), (tex_map,) = generated_mesh
    savemeshtes2(
        mesh_v.data.cpu().numpy(),
        all_uvs.data.cpu().numpy(),
        mesh_f.data.cpu().numpy(),
        all_mesh_tex_idx.data.cpu().numpy(),
        os.path.join(mesh_dir, '%07d.obj' % (save_mesh_idx))
    )

    lo, hi = (-1, 1)
    img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = img.clip(0, 255)
    mask = np.sum(img.astype(float), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(float)
    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    img = img * (1 - mask) + dilate_img * mask
    img = img.clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
        os.path.join(mesh_dir, '%07d.png' % (save_mesh_idx)))
    print(save_mesh_idx)
    save_mesh_idx += 1
