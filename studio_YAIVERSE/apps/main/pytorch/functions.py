import os
import io
import tempfile
import zipfile
import PIL.Image
import numpy as np
import cv2
import torch
import trimesh

from .nn import get_generator_ema, get_device

from typing import TYPE_CHECKING, overload
if TYPE_CHECKING:
    from typing import *


def format_material(filename: "str") -> "str":
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


def format_mesh_obj(
        pointnp_px3: "Union[np.ndarray, torch.Tensor]",
        tcoords_px2: "Union[np.ndarray, torch.Tensor]",
        facenp_fx3: "Union[np.ndarray, torch.Tensor]",
        facetex_fx3: "Union[np.ndarray, torch.Tensor]",
        filename: "str"
) -> "str":
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


def postprocess_texture_map(tensor: "torch.Tensor") -> "PIL.Image.Image":
    lo, hi = -1, 1
    tensor = (tensor - lo) * (255 / (hi - lo))
    tensor = tensor.clip(0, 255).float()
    img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    mask = np.sum(img.astype(float), axis=-1, keepdims=True)
    mask = (mask <= 3.0).astype(float)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilate_img = cv2.dilate(img, kernel, iterations=1)  # NOQA
    img = img * (1 - mask) + dilate_img * mask
    img = img.clip(0, 255).astype(np.uint8)
    return PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB')


@overload
def inference(
        name: "str",
        text: "Optional[str]" = None,
        extensions: "Sequence[str]" = ("glb", "png")
) -> "Dict[str, io.BytesIO]": ...


@overload
def inference(
        name: "str",
        text: "Optional[str]" = None,
        extensions: "Sequence[()]" = ("glb", "png")
) -> "io.BytesIO": ...


@torch.inference_mode()
def inference(name, text=None, extensions=("glb", "png")):

    print("Running inference(name={name!r}, text={text!r})".format(name=name, text=text))

    device = get_device()
    g_ema = get_generator_ema()

    geo_z = torch.randn([1, g_ema.z_dim], device=device)
    tex_z = torch.randn([1, g_ema.z_dim], device=device)

    c_to_compute_w_avg = None
    g_ema.update_w_avg(c_to_compute_w_avg)

    generated_mesh = g_ema.generate_3d_mesh(
        geo_z=geo_z, tex_z=tex_z, c=None, truncation_psi=0.7,
        use_style_mixing=False
    )
    (mesh_v,), (mesh_f,), (all_uvs,), (all_mesh_tex_idx,), (tex_map,) = generated_mesh

    mesh_obj = format_mesh_obj(
        mesh_v.data.cpu().numpy(),
        all_uvs.data.cpu().numpy(),
        mesh_f.data.cpu().numpy(),
        all_mesh_tex_idx.data.cpu().numpy(),
        name
    )
    material = format_material(name)
    texture_map = postprocess_texture_map(tex_map)

    with tempfile.TemporaryDirectory() as tempdir:

        mesh_obj_name = os.path.join(tempdir, name + '.obj')
        with open(mesh_obj_name, 'w') as fp:
            fp.write(mesh_obj)
        material_name = os.path.join(tempdir, name + '.mtl')
        with open(material_name, 'w') as fp:
            fp.write(material)
        text_map_name = os.path.join(tempdir, name + '.png')
        with open(text_map_name, 'wb') as fp:
            texture_map.save(fp)

        result = {}

        if extensions:
            mesh = trimesh.load("{}.obj".format(mesh_obj_name))
            for ext in extensions:
                if ext.lower() == "png":
                    result[ext] = io.BytesIO(mesh.scene().save_image(visible=False, resolution=(512, 512)))
                else:
                    result[ext] = io.BytesIO(mesh.export(file_type=ext))
        else:
            bundle_fp = io.BytesIO()
            bundle = zipfile.ZipFile(bundle_fp, "w")
            bundle.write(mesh_obj_name, arcname=name + '.obj', compress_type=zipfile.ZIP_DEFLATED)
            bundle.write(material_name, arcname=name + '.mtl', compress_type=zipfile.ZIP_DEFLATED)
            bundle.write(text_map_name, arcname=name + '.png', compress_type=zipfile.ZIP_DEFLATED)
            return bundle_fp

        return result
