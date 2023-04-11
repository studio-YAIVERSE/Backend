import os

import torch
from training.inference_utils import (
    save_visualization, save_visualization_for_interpolation,
    save_textured_mesh_for_inference, save_geo_for_inference
)

from django.conf import settings
from rest_framework.exceptions import PermissionDenied

from ..pytorch import get_generator_ema


@torch.inference_mode(True)
def inference(
        run_dir='.',  # Output directory.
        inference_to_generate_textured_mesh=False,
        inference_save_interpolation=False,
        inference_generate_geo=False
):
    if not settings.TORCH_ENABLED:
        raise PermissionDenied("Server started without pytorch initialization.")

    device = torch.device(settings.DEVICE)
    G_ema = get_generator_ema()

    grid_size = (5, 5)
    n_shape = grid_size[0] * grid_size[1]
    grid_z = torch.randn([n_shape, G_ema.z_dim], device=device).split(1)  # random code for geometry
    grid_tex_z = torch.randn([n_shape, G_ema.z_dim], device=device).split(1)  # random code for texture
    grid_c = torch.ones(n_shape, device=device).split(1)

    print('==> generate ')
    save_visualization(
        G_ema, grid_z, grid_c, run_dir, 0, grid_size, 0,
        save_all=False,
        grid_tex_z=grid_tex_z
    )

    if inference_to_generate_textured_mesh:
        print('==> generate inference 3d shapes with texture')
        save_textured_mesh_for_inference(
            G_ema, grid_z, grid_c, run_dir,
            save_mesh_dir='texture_mesh_for_inference',
            c_to_compute_w_avg=None,
            grid_tex_z=grid_tex_z
        )

    if inference_save_interpolation:
        print('==> generate interpolation results')
        save_visualization_for_interpolation(G_ema, save_dir=os.path.join(run_dir, 'interpolation'))

    if inference_generate_geo:
        print('==> generate 7500 shapes for evaluation')
        save_geo_for_inference(G_ema, run_dir)
