import os
import torch
import numpy as np
import trimesh
import argparse
import pyvista
from mesh_to_sdf import mesh_to_sdf

from funcwm.cu3d import sdf_by_normal

BBOX = 1.0


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0, keepdims=True)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=-1))) * BBOX
    pc = pc / m
    return pc, centroid, m



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--nsample", type=int, default=500000)
    args = parser.parse_args()
    visual = False

    name = os.path.basename(os.path.splitext(args.model)[0])
    nsample = args.nsample
    if os.path.splitext(args.model)[1] == ".npy":
        checkpoint = np.load(args.model, allow_pickle=True).item()
        mesh = checkpoint["mesh"]
        mesh.vertices = pc_normalize(mesh.vertices)[0]
        # mesh.show()
        print(mesh)
    else:
        mesh = trimesh.load_mesh(args.model)
        # tmp, center, m = pc_normalize(mesh.vertices)
        # print(tmp, center, m)
        mesh.vertices = pc_normalize(mesh.vertices)[0]

    on_surface_x, on_surface_face = mesh.sample(nsample, return_index=True)
    on_surface_x = on_surface_x.astype(np.float32)
    on_surface_normal = mesh.face_normals[on_surface_face]
    on_surface_sdf = mesh_to_sdf(mesh, on_surface_x, 
        surface_point_method='scan',
        sign_method='normal',
        bounding_radius=None,
        scan_count=100,
        scan_resolution=400,
        normal_sample_count=100).reshape(-1, 1)

    off_surface_x = np.random.uniform(-1, 1, size=(nsample, 3)).astype(np.float32)
    off_surface_normal = np.ones(off_surface_x.shape, dtype=np.float32) * -1
    # off_surface_sdf = sdf_by_normal(
    #     torch.FloatTensor(mesh.vertices).cuda().unsqueeze(0).contiguous(),
    #     torch.FloatTensor(mesh.vertex_normals).cuda().unsqueeze(0).contiguous(),
    #     torch.FloatTensor(off_surface_x).cuda().unsqueeze(0).contiguous(),
    # ).reshape(-1, 1).cpu().numpy()
    off_surface_sdf = mesh_to_sdf(mesh, off_surface_x, 
        surface_point_method='scan',
        sign_method='normal',
        bounding_radius=None,
        scan_count=100,
        scan_resolution=400,
        normal_sample_count=100).reshape(-1, 1)

    if visual:
        print((on_surface_sdf < 0).sum())
        mesh = pyvista.PolyData(on_surface_x)
        colors = np.zeros(on_surface_x.shape)
        colors[on_surface_sdf < 0, 2] = 1
        colors[on_surface_sdf > 0, 0] = 5
        mesh["point_color"] = colors
        mesh["Normals"] = on_surface_normal
        p = pyvista.Plotter()
        p.add_mesh(mesh, scalars="point_color")
        p.add_mesh(mesh.glyph(geom=pyvista.Arrow(tip_length=0.01, tip_radius=0.01, shaft_radius=0.005, scale=0.01), orient="Normals"), color="black")
        p.show()
    
    sdfs = np.concatenate((on_surface_sdf, off_surface_sdf), axis=0)
    points = np.concatenate((on_surface_x, off_surface_x), axis=0)
    normals = np.concatenate((on_surface_normal, off_surface_normal), axis=0)
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, name), {
        "name": name,
        "points": points,
        "sdfs": sdfs,
        "normals": normals,
    })