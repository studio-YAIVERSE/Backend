import io
import PIL.Image
import numpy as np
import cv2
import torch


from .nn import get_generator_ema, get_device


def format_material(filename):
    material = (
        'newmtl material_0\n'
        'Kd 1 1 1\n'
        'Ka 0 0 0\n'
        'Ks 0.4 0.4 0.4\n'
        'Ns 10\n'
        'illum 2\n'
        'map_Kd {filename}.png\n'
    )
    return material.format(filename=filename)


def format_mesh_obj(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, filename):
    buf = io.StringIO()
    try:
        buf.write('mtllib {filename}.mtl\n'.format(filename=filename))
        for _, pp in enumerate(pointnp_px3):
            buf.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))
        for _, pp in enumerate(tcoords_px2):
            buf.write('vt %f %f\n' % (pp[0], pp[1]))
        buf.write('usemtl material_0\n')
        for i, f in enumerate(facenp_fx3):
            f1 = f + 1
            f2 = facetex_fx3[i] + 1
            buf.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
        return buf.getvalue()
    finally:
        buf.close()


def postprocess_texture_map(array):
    lo, hi = (-1, 1)
    img = np.asarray(array.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = img.clip(0, 255)
    mask = np.sum(img.astype(float), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(float)
    kernel = np.ones((3, 3), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    img = img * (1 - mask) + dilate_img * mask
    img = img.clip(0, 255).astype(np.uint8)
    return PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB')


@torch.inference_mode()
def inference(filename):
    # from rest_framework.exceptions import PermissionDenied
    # if not settings.TORCH_ENABLED:
    #     raise PermissionDenied("Server started without pytorch initialization.")

    device = get_device()
    G_ema = get_generator_ema()

    geo_z = torch.randn([1, G_ema.z_dim], device=device)
    tex_z = torch.randn([1, G_ema.z_dim], device=device)

    c_to_compute_w_avg = None
    G_ema.update_w_avg(c_to_compute_w_avg)

    generated_mesh = G_ema.generate_3d_mesh(
        geo_z=geo_z, tex_z=tex_z, c=None, truncation_psi=0.7,
        use_style_mixing=False
    )
    (mesh_v,), (mesh_f,), (all_uvs,), (all_mesh_tex_idx,), (tex_map,) = generated_mesh

    mesh_obj = format_mesh_obj(
        mesh_v.data.cpu().numpy(),
        all_uvs.data.cpu().numpy(),
        mesh_f.data.cpu().numpy(),
        all_mesh_tex_idx.data.cpu().numpy(),
        filename
    )
    material = format_material(filename)
    texture_map = postprocess_texture_map(tex_map)

    with open(filename + '.obj', 'w') as fp:
        fp.write(mesh_obj)
    with open(filename + '.mtl', 'w') as fp:
        fp.write(material)
    with open(filename + '.png', 'wb') as fp:
        texture_map.save(fp)
