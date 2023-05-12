import os
import io
import contextlib
import tempfile
import PIL.Image
import numpy as np
import cv2
import torch
import nvdiffrast.torch as dr
import trimesh
import trimesh.proximity
import trimesh.voxel.ops
import pyglet
import xvfbwrapper
from operator import itemgetter

from .utils import inference_result

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from .nn import CLIPLoss
    from training.networks_get3d import GeneratorDMTETMesh


def multibox(centers, pitch=1.0, colors=None):  # function for voxelize()
    from trimesh import primitives
    from trimesh.base import Trimesh
    # get centers as numpy array
    centers = np.asanyarray(
        centers, dtype=np.float64)
    # get a basic box
    b = primitives.Box()
    # apply the pitch
    b.apply_scale(float(pitch))
    # tile into one box vertex per center
    v = np.tile(
        centers,
        (1, len(b.vertices))).reshape((-1, 3))
    # offset to centers
    v += np.tile(b.vertices, (len(centers), 1))

    f = np.tile(b.faces, (len(centers), 1))
    f += np.tile(
        np.arange(len(centers)) * len(b.vertices),
        (len(b.faces), 1)).T.reshape((-1, 1))
    face_colors = colors.repeat(12, axis=0)
    vertex_colors = colors.repeat(8, axis=0)

    return Trimesh(
        vertices=v,
        faces=f,
        face_colors=face_colors,
        vertex_colors=vertex_colors,
        validate=True,
        merge_tex=True,
        merge_norm=True,
    )


def voxelize(mesh: trimesh.Trimesh, voxel_size: float = 0.01, quantize: bool = True):
    angel_voxel = mesh.voxelized(voxel_size).hollow()
    colors = np.asarray(mesh.visual.to_color().vertex_colors)
    if quantize:
        quantize_factor = 41
        colors = (np.floor(colors.astype(float) / quantize_factor) * quantize_factor).astype(np.uint8)
    mesh_verts = mesh.vertices
    _, vert_idx = trimesh.proximity.ProximityQuery(mesh).vertex(angel_voxel.points)
    cube_color = np.zeros([angel_voxel.shape[0], angel_voxel.shape[1], angel_voxel.shape[2], 4], dtype=np.uint8)
    for idx, vert in enumerate(vert_idx):
        vox_verts = angel_voxel.points_to_indices(mesh_verts[vert])
        curr_color = colors[vert]
        curr_color[3] = 255
        cube_color[vox_verts[0], vox_verts[1], vox_verts[2], :] = curr_color
    voxelized_mesh = multibox(
        angel_voxel.sparse_indices.astype(float),
        colors=cube_color[angel_voxel.encoding.dense]
    )
    voxelized_mesh.apply_transform(angel_voxel.transform)
    return voxelized_mesh


def convert_obj_to_extension(
        name: "str",
        mesh_obj: "str",
        material: "Optional[str]" = None,
        texture_map: "Optional[PIL.Image.Image]" = None,
        tempdir: "Optional[str]" = None
) -> "Tuple[io.BytesIO, trimesh.Trimesh]":

    with contextlib.ExitStack() as stk:
        if tempdir is None:
            tempdir = stk.enter_context(tempfile.TemporaryDirectory())

        mesh_obj_name = os.path.join(tempdir, name + '.obj')
        with open(mesh_obj_name, 'w') as fp:
            fp.write(mesh_obj)
        material_name = os.path.join(tempdir, name + '.mtl')
        if material is not None:
            with open(material_name, 'w') as fp:
                fp.write(material)
        text_map_name = os.path.join(tempdir, name + '.png')
        if texture_map is not None:
            with open(text_map_name, 'wb') as fp:
                texture_map.save(fp)

        mesh = trimesh.load(mesh_obj_name)
        file = io.BytesIO(mesh.export(file_type="glb"))

        return file, mesh


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


def thumbnail_to_pil(tensor: "torch.Tensor") -> "PIL.Image.Image":
    lo, hi = -1, 1
    N, C, H, W = tensor.shape  # noqa
    assert N == 1 and C == 3
    tensor = (tensor - lo) * (255 / (hi - lo))
    img = tensor.detach().cpu().numpy().astype(np.float32)
    img = np.rint(img).clip(0, 255).astype(np.uint8).squeeze(0).transpose(1, 2, 0)
    return PIL.Image.fromarray(img, 'RGB')


def postprocess_outputs(generated_thumbnail, generated_mesh, name: "str"):
    img, _ = generated_thumbnail
    thumbnail = io.BytesIO()
    thumbnail_to_pil(img[:, :3]).save(thumbnail, format="PNG")

    (mesh_v,), (mesh_f,), (all_uvs,), (all_mesh_tex_idx,), (tex_map,) = generated_mesh
    mesh_str = format_mesh_obj(
        mesh_v.data.cpu().numpy(),
        all_uvs.data.cpu().numpy(),
        mesh_f.data.cpu().numpy(),
        all_mesh_tex_idx.data.cpu().numpy(),
        name
    )
    mtl_str = format_material(name)
    tex_bin = postprocess_texture_map(tex_map)

    with tempfile.TemporaryDirectory() as tmpdir:
        file, mesh = convert_obj_to_extension(name, mesh_str, mtl_str, tex_bin, tmpdir)

    vm = voxelize(mesh)
    vf = io.BytesIO(vm.export(file_type="glb"))

    try:
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(xvfbwrapper.Xvfb())
            except OSError:
                pass
            window_conf = pyglet.gl.Config(double_buffer=True, depth_size=24)
            img = vm.scene().save_image(resolution=(512, 512), visible=False, window_conf=window_conf)
    except Exception as exc:
        from .utils import log_pytorch
        log_pytorch("Failed to render image - %s: %s" % (type(exc).__name__, exc), level=2)
        from .fallback import _fallback_image
        img = _fallback_image()

    vt = io.BytesIO(img)

    return inference_result(file=file, thumbnail=thumbnail, voxelized_file=vf, voxelized_thumbnail=vt)


def with_log(obj: "Any") -> "Any":
    import pprint
    from .utils import should_log, log_pytorch
    if should_log(level=3):
        output = io.StringIO()
        try:
            obj = list(obj)
        except (TypeError, ValueError):
            pass
        try:
            pprint.pprint(dict(obj), stream=output)
        except (TypeError, ValueError):
            pprint.pprint(obj, stream=output)
        log_pytorch(output.getvalue(), level=3)
    return obj


def cosine_distance(src: "torch.Tensor", trg: "torch.Tensor") -> "float":
    return torch.cosine_similarity(src, trg).mean().neg_().add_(1.).item()


def schedule_nada_training(target, feat):  # noqa
    # TODO: Implement NADA training
    # In this MVP version, we actually don't re-train NADA network,
    # but we can do it in the future by additional implementation.
    pass


def map_checkpoint(clip_loss: "CLIPLoss", clip_map: "dict", target: "Union[str, io.BytesIO]") -> "Tuple[str, str]":
    if isinstance(target, io.BytesIO):
        feat = clip_loss.preprocessing_image(target)
    elif isinstance(target, str):
        feat = clip_loss.get_text_features(target)
    else:
        raise TypeError(target)
    key_src = min(
        with_log((k, cosine_distance(v[0], feat)) for k, v in clip_map.items()),
        key=itemgetter(1))[0]
    key_dsts = sorted(
        with_log((k, cosine_distance(v, feat)) for k, v in clip_map[key_src][1].items()),
        key=itemgetter(1))[:2]
    if key_dsts[0][0] == key_src and key_dsts[0][1] > .1:
        key_dst, loss = key_dsts[1]
    else:
        key_dst, loss = key_dsts[0]
    if loss > .19:
        schedule_nada_training(target, feat)
    return key_src, key_dst


def generate_latent(
        g_ema: "GeneratorDMTETMesh",
        generator: "Optional[torch.Generator]" = None
) -> "torch.Tensor":
    return torch.randn([1, g_ema.z_dim], device=g_ema.device, generator=generator)


def inference_logic(
        g_ema: "GeneratorDMTETMesh",
        geo_z,
        tex_z,
        c=None,
        truncation_psi=1.,
        truncation_cutoff=None,
        camera=None,                            # -> g_ema.generate_3d (synthesis.generate)
        update_emas=False, use_mapping=True,    # -> g_ema.generate_3d_mesh
        texture_resolution=2048,                # -> synthesis.extract_3d_shape
):
    """
    Description :
    To return [3D mesh, mtl, rendered img] , mix 'def generate_3d()' & 'def generate_3d_mesh()'
    This is same as mixture of 'def generate()' and 'def extract_3d_shape()'

    Note :
    we don't take below as input args
        1. use_style_mixing
        2. generate_no_light
        3. with_texture
        , since they are redundant.

    Return :
        return_generate_3d = [rendered RGB Image, rendered 2D Silhouette image]
        return_generate_3d_mesh = [mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, texture map]
    """

    if use_mapping:
        ws = g_ema.mapping(
            tex_z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            update_emas=update_emas)
        ws_geo = g_ema.mapping_geo(
            geo_z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            update_emas=update_emas)
    else:
        ws = tex_z
        ws_geo = geo_z

    self = g_ema.synthesis
    tex_feature = ...

    # ----------------------- synthesis.generate ----------------------- #

    # (1) Generate 3D mesh first
    # NOTE :
    # this code is shared by 'def generate' and 'def extract_3d_mesh'
    if self.one_3d_generator:
        sdf_feature, tex_feature = self.generator.get_feature(
            ws[:, :self.generator.tri_plane_synthesis.num_ws_tex],
            ws_geo[:, :self.generator.tri_plane_synthesis.num_ws_geo])
        ws = ws[:, self.generator.tri_plane_synthesis.num_ws_tex:]
        ws_geo = ws_geo[:, self.generator.tri_plane_synthesis.num_ws_geo:]
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo, sdf_feature)
    else:
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo)

    ws_tex = ws

    # (2) Generate random camera
    if camera is None:
        campos, cam_mv, rotation_angle, elevation_angle, sample_r = self.generate_random_camera(
            ws_tex.shape[0], n_views=self.n_views)
        gen_camera = (campos, cam_mv, sample_r, rotation_angle, elevation_angle)
        run_n_view = self.n_views
    else:
        if isinstance(camera, tuple):
            cam_mv = camera[0]
            campos = camera[1]
        else:
            cam_mv = camera
            campos = None
        gen_camera = camera
        run_n_view = cam_mv.shape[1]

    # (3) Render the mesh into 2D image (get 3d position of each image plane)
    antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)

    tex_pos = return_value['tex_pos']
    tex_hard_mask = hard_mask
    tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
    tex_hard_mask = torch.cat(
        [torch.cat(
            [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
             for i_view in range(run_n_view)], dim=2)
            for i in range(ws_tex.shape[0])], dim=0)

    # (4) Querying the texture field to predict the texture feature for each pixel on the image
    if self.one_3d_generator:
        tex_feat = self.get_texture_prediction(
            ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask,
            tex_feature)
    else:
        tex_feat = self.get_texture_prediction(
            ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask)
    background_feature = torch.zeros_like(tex_feat)

    # (5) Merge them together
    img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

    # (6) We should split it back to the original image shape
    img_feat = torch.cat(
        [torch.cat(
            [img_feat[i:i + 1, :, self.img_resolution * i_view: self.img_resolution * (i_view + 1)]
             for i_view in range(run_n_view)], dim=0) for i in range(len(return_value['tex_pos']))], dim=0)

    ws_list = [ws_tex[i].unsqueeze(dim=0).expand(return_value['tex_pos'][i].shape[0], -1, -1) for i in
               range(len(return_value['tex_pos']))]
    ws = torch.cat(ws_list, dim=0).contiguous()

    # (7) Predict the RGB color for each pixel (self.to_rgb is 1x1 convolution)
    if self.feat_channel > 3:
        network_out = self.to_rgb(img_feat.permute(0, 3, 1, 2), ws[:, -1])
    else:
        network_out = img_feat.permute(0, 3, 1, 2)
    img = network_out

    if self.render_type == 'neural_render':
        img = img[:, :3]
    else:
        raise NotImplementedError

    # print('before concat img : ', img.shape) - (1, 3, 1024 ,1024)
    img = torch.cat([img, antilias_mask.permute(0, 3, 1, 2)], dim=1)

    yield img, antilias_mask

    del tex_hard_mask
    del tex_feat

    # ------------------- synthesis.extract_3d_shape ------------------- #

    # (8) Use x-atlas to get uv mapping for the mesh
    from training.extract_texture_map import xatlas_uvmap
    all_uvs = []
    all_mesh_tex_idx = []
    all_gb_pose = []
    all_uv_mask = []
    if self.dmtet_geometry.renderer.ctx is None:
        self.dmtet_geometry.renderer.ctx = dr.RasterizeGLContext(device=self.device)
    for v, f in zip(mesh_v, mesh_f):
        uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap(
            self.dmtet_geometry.renderer.ctx, v, f, resolution=texture_resolution)
        all_uvs.append(uvs)
        all_mesh_tex_idx.append(mesh_tex_idx)
        all_gb_pose.append(gb_pos)
        all_uv_mask.append(mask)

    tex_hard_mask = torch.cat(all_uv_mask, dim=0).float()

    # (9) Query the texture field to get the RGB color for texture map
    all_network_output = []
    for _ws, _all_gb_pose, _ws_geo, _tex_hard_mask in zip(ws, all_gb_pose, ws_geo, tex_hard_mask):
        if self.one_3d_generator:
            tex_feat = self.get_texture_prediction(
                _ws.unsqueeze(dim=0), [_all_gb_pose],
                _ws_geo.unsqueeze(dim=0).detach(),
                _tex_hard_mask.unsqueeze(dim=0),
                tex_feature)
        else:
            tex_feat = self.get_texture_prediction(
                _ws.unsqueeze(dim=0), [_all_gb_pose],
                _ws_geo.unsqueeze(dim=0).detach(),
                _tex_hard_mask.unsqueeze(dim=0))
        background_feature = torch.zeros_like(tex_feat)
        # Merge them together
        img_feat = tex_feat * _tex_hard_mask.unsqueeze(dim=0) + background_feature * (
                1 - _tex_hard_mask.unsqueeze(dim=0))
        network_out = self.to_rgb(img_feat.permute(0, 3, 1, 2), _ws.unsqueeze(dim=0)[:, -1])
        all_network_output.append(network_out)
    network_out = torch.cat(all_network_output, dim=0)

    yield mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out
