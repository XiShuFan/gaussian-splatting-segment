#!/usr/bin/env python
# 通用 GLB 渲染器 - 支持 EGL / OSMesa 回退，无头服务器可用

import os
import sys
import json
import numpy as np
import trimesh
import imageio

# ================================
# 🔧 第一步：设置 OpenGL 平台（必须在导入 pyrender 之前！）
# ================================

# 尝试优先使用 EGL（GPU 加速）
try:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    from OpenGL import EGL
    print("🟢 Using EGL platform (GPU-accelerated rendering)")
except ImportError:
    print("🟡 EGL not available, falling back to OSMesa...")
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

# 现在导入 pyrender（和其他）
import pyrender
from pyrender.constants import RenderFlags
from pyrender import MetallicRoughnessMaterial, Texture, Primitive, Mesh as PyMesh
from trimesh.visual import TextureVisuals
from PIL import Image
from render_one_camera import sample_cameras_on_circle


# ================================
# 🌐 在单位球面上均匀采样 n 个点
# ================================
def fibonacci_sphere(n):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # 黄金角

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)


# ================================
# 🌐 经度×纬度 规则球面采样
# ================================
def latlon_sphere(n_lon=8):
    """
    n_lat: 纬度方向采样数
    n_lon: 经度方向采样数
    return: (n_lat * n_lon, 3)
    """
    points = []

    # 只有0的效果是正确的
    latitudes = [0]
    # 经度：[0, 2pi)
    longitudes = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    
    for phi in latitudes:
        for lam in longitudes:     # 经度
            x = np.cos(phi) * np.cos(lam)
            y = np.sin(phi)
            z = np.cos(phi) * np.sin(lam)
            points.append([x, y, z])

    return np.array(points)


def check_glb(mesh):
    print("Loaded mesh:", type(mesh), mesh)

    if isinstance(mesh, trimesh.Scene):
        print(f"Scene contains {len(mesh.geometry)} geometries:")
        
        for i, (name, geo) in enumerate(mesh.geometry.items()):
            print(f"  [{i}] Geometry name: {name}")
            print(f"      Type: {type(geo)}")
            
            if hasattr(geo, 'visual') and geo.visual is not None:
                mat = geo.visual.material
                print(f"      Material type: {type(mat)}")
                
                # 检查是否有 baseColorTexture
                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    print("      ✅ Has base color texture!")
                elif hasattr(mat, 'image') and mat.image is not None:
                    print("      ✅ Has embedded image (likely PNG/JPG)")
                else:
                    print("      ❌ No texture found in this geometry")
                    
                # 打印常用属性帮助调试
                try:
                    print("      Scalar properties:", {
                        k: getattr(mat, k)
                        for k in ['metallicFactor', 'roughnessFactor', 'alphaMode']
                        if hasattr(mat, k)
                    })
                except:
                    pass
            else:
                print("      ❌ No visual/material info")
    else:
        # 如果是单个 mesh
        print("Single mesh detected")
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            mat = mesh.visual.material
            print("Material type:", type(mat))
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                print("✅ Found base color texture!")
            elif hasattr(mat, 'image') and mat.image is not None:
                print("✅ Found image in material")
            else:
                print("❌ No texture in mesh")
        else:
            print("❌ No visual data")


def pil_to_rgb(img: Image.Image) -> Image.Image:
    """把 RGBA/L/LA 等统一变为 RGB（避免 alpha 导致发黑/异常）"""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")

def build_pyrender_mesh_from_trimesh(geo: trimesh.Trimesh):
    """
    最稳的方式：手动构造 Primitive，显式传 positions/normals/uv/indices。
    """
    # 顶点/面
    positions = np.asarray(geo.vertices, dtype=np.float32)
    indices = np.asarray(geo.faces, dtype=np.int32)

    # 法线（没有就算）
    normals = None
    try:
        if geo.vertex_normals is not None and len(geo.vertex_normals) == len(geo.vertices):
            normals = np.asarray(geo.vertex_normals, dtype=np.float32)
    except Exception:
        normals = None

    # UV + 纹理
    texcoords = None
    material = None

    if hasattr(geo, "visual") and isinstance(geo.visual, TextureVisuals):
        # UV
        if getattr(geo.visual, "uv", None) is not None:
            texcoords = np.asarray(geo.visual.uv, dtype=np.float32)

        # 纹理图：trimesh 最常见的入口是 material.image
        img = None
        mat0 = getattr(geo.visual, "material", None)
        if mat0 is not None:
            img = getattr(mat0, "image", None)

            # 有些 glTF/PBR 材质在 trimesh 里也可能挂在 baseColorTexture 上
            # 但类型不一定是 PIL.Image，这里做个兜底尝试
            if img is None and hasattr(mat0, "baseColorTexture"):
                img = mat0.baseColorTexture

        if isinstance(img, Image.Image):
            img = pil_to_rgb(img)
            tex = Texture(source=img, source_channels="RGB")
            material = MetallicRoughnessMaterial(
                baseColorTexture=tex,
                metallicFactor=0.0,
                roughnessFactor=1.0,
                alphaMode="OPAQUE"
            )

    # 没纹理 or UV 缺失：用一个非灰的默认色，方便你肉眼判断
    if material is None:
        material = MetallicRoughnessMaterial(
            baseColorFactor=[0.9, 0.9, 0.9, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )

    prim = Primitive(
        positions=positions,
        indices=indices,
        normals=normals,
        texcoord_0=texcoords,
        material=material
    )
    return PyMesh([prim])



# ================================
# 🎬 主函数：加载 GLB -> 渲染 -> 输出图像和相机参数
# ================================
def render_glb_universal(
    glb_path,
    output_dir,
    n_cameras,
    img_width,
    img_height,
    sphere_scale  # 相机距离 = 包围球半径 * scale
):
    # 检查输入文件
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    # -------------------------------
    # 1. 加载并处理模型
    # -------------------------------
    mesh = trimesh.load(glb_path, force='scene')

    check_glb(mesh)

    if not isinstance(mesh, trimesh.Scene):
        raise ValueError("Expected a GLB file containing a scene")

    print(f"Loaded a scene with {len(mesh.geometry)} geometries.")

    # 提取所有顶点用于包围球计算
    verts_world = []
    for name, geo in mesh.geometry.items():
        if not isinstance(geo, trimesh.Trimesh):
            continue
        transform = mesh.graph.get(name)[0]
        vertices_transformed = trimesh.transform_points(geo.vertices, transform)
        verts_world.append(vertices_transformed)

    if len(verts_world) == 0:
        raise ValueError("No valid mesh geometry found")

    verts_world = np.vstack(verts_world)
    center = verts_world.mean(axis=0)
    radius = np.max(np.linalg.norm(verts_world - center, axis=1))
    print(f"Bounding sphere: center={center}, radius={radius:.4f}")


    # -------------------------------
    # 2. 生成相机位置（在包围球外）
    # -------------------------------
    points_on_unit_sphere = latlon_sphere() # fibonacci_sphere(n_cameras)
    camera_positions = center + radius * sphere_scale * points_on_unit_sphere

    # -------------------------------
    # 3. 创建 PyRender 场景 + 添加所有带材质的子网格（修正版：确保 UV + image 正确传递）
    # -------------------------------
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])

    # 遍历 scene 的 geometry，按 node 取 pose，确保每个 node 的变换正确
    for node_name in mesh.graph.nodes:
        # graph.get(node_name) -> (matrix, geometry_name)
        try:
            mat4, geom_name = mesh.graph.get(node_name)
        except Exception:
            continue

        if geom_name is None or geom_name not in mesh.geometry:
            continue

        geo = mesh.geometry[geom_name]
        if not isinstance(geo, trimesh.Trimesh):
            continue
        print(type(geo.visual.material), geo.visual.material)
        print("has uv", geo.visual.uv is not None, "uv shape", None if geo.visual.uv is None else geo.visual.uv.shape)
        print("image type", type(getattr(geo.visual.material, "image", None)))

        try:
            py_mesh = build_pyrender_mesh_from_trimesh(geo)
            scene.add(py_mesh, name=f"{node_name}:{geom_name}", pose=mat4)
            print(f"✔️ Added '{geom_name}' (node '{node_name}') with texture={isinstance(getattr(getattr(geo.visual,'material',None),'image',None), Image.Image)}")
        except Exception as e:
            print(f"❌ Failed to add geometry '{geom_name}' from node '{node_name}': {e}")

    # 相机内参
    fov_x_deg = 55.0
    fov_x = np.deg2rad(fov_x_deg)

    fx = (img_width / 2.0) / np.tan(fov_x / 2.0)
    fy = fx  # 方形像素，通常这样设置更自然
    cx = img_width / 2.0
    cy = img_height / 2.0

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    camera_node = scene.add(camera)

    # -------------------------------
    # 4. 初始化离屏渲染器
    # -------------------------------
    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
        print(f"✅ Offscreen renderer created ({img_width}x{img_height})")
    except Exception as e:
        print(f"❌ Failed to create renderer: {e}")
        sys.exit(1)

    # -------------------------------
    # 5. 遍历每个相机位置进行渲染
    # -------------------------------
    cameras_data = []

    for i, pos in enumerate(camera_positions):
        # --- 设置相机 pose（含正确朝向）---
        up = np.array([0, 1, 0])
        z_axis = pos - center
        z_axis /= np.linalg.norm(z_axis)

        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([1, 0, 0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        R_c2w = np.column_stack((x_axis, y_axis, z_axis))  # 列为轴

        # 添加相机参数
        flip_z = np.diag([1, -1, -1])
        R_c2w_3dgs = R_c2w @ flip_z
        R_w2c = R_c2w_3dgs.T

        cameras_data.append({
            "id": i,
            "img_name": f"frame_{i:05d}.png",
            "width": img_width,
            "height": img_height,
            "position": pos.tolist(),
            "rotation": R_w2c.tolist(),
            "fy": fy,
            "fx": fx
        })
    
    # 扩展采样
    all_cameras = [] + cameras_data
    all_new_cams = []
    for cam in cameras_data:
        new_cams = sample_cameras_on_circle(cam, center, radius=radius, n=6)
        all_new_cams += new_cams
        new_cams = sample_cameras_on_circle(cam, center, radius=radius * 2, n=6)
        all_new_cams += new_cams
        new_cams = sample_cameras_on_circle(cam, center, radius=radius * 3, n=6)
        all_new_cams += new_cams
        new_cams = sample_cameras_on_circle(cam, center, radius=radius * 6, n=6)
        all_new_cams += new_cams
    for cam in all_new_cams:
        # 设置中间点
        cam["position"] = ((np.asarray(cam["position"]) + center) / 2).tolist()
    all_cameras += all_new_cams
    for i, cam in enumerate(all_cameras):
        # --- 设置相机 pose（含正确朝向）---
        pos = np.asarray(cam["position"])
        up = np.array([0, 1, 0])
        z_axis = pos - center
        z_axis /= np.linalg.norm(z_axis)

        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([1, 0, 0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        R_c2w = np.column_stack((x_axis, y_axis, z_axis))  # 列为轴
        T = np.eye(4)
        T[:3, :3] = R_c2w
        T[:3, 3] = pos
        scene.set_pose(camera_node, T)

        # --- 渲染图像 ---
        try:
            flags = RenderFlags.RGBA | RenderFlags.FLAT
            color, _ = renderer.render(scene, flags=flags)
            img_path = os.path.join(image_output_dir, f"frame_{i:05d}.png")
            imageio.imwrite(img_path, color)
            print(f"📸 Rendered: {img_path}")
        except Exception as e:
            print(f"⚠️ Rendering failed for camera {i}: {e}")
        
        cam["img_name"] = f"frame_{i:05d}.png"
        cam["position"] = [-pos[0], -pos[1], pos[2]]
        flip_xy = np.diag([-1.0, -1.0, 1.0])
        cam["rotation"] = (flip_xy @ np.asarray(cam["rotation"])).tolist()
        

    # -------------------------------
    # 6. 保存相机参数
    # -------------------------------
    json_path = os.path.join(output_dir, "cameras.json")
    with open(json_path, 'w') as f:
        json.dump(all_cameras, f, indent=2)
    print(f"💾 Camera parameters saved to: {json_path}")

    # -------------------------------
    # 7. 释放资源
    # -------------------------------
    renderer.delete()


# ================================
# 🚀 主程序入口
# ================================
if __name__ == "__main__":
    PROJECT = "shoes"
    BASE_FOLDER = f"/media/why/新加卷/xsf/商品3DGS/mesh_to_gs/{PROJECT}/"
    # 默认参数
    input_glb = os.path.join(BASE_FOLDER, f"{PROJECT}.glb")
    output_dir = os.path.join(BASE_FOLDER, "render_output")

    if len(sys.argv) >= 3:
        input_glb = sys.argv[1]
        output_dir = sys.argv[2]

    print(f"🎯 Input GLB: {input_glb}")
    print(f"📁 Output dir: {output_dir}")

    render_glb_universal(
        glb_path=input_glb,
        output_dir=output_dir,
        n_cameras=5,
        img_width=1024,
        img_height=1024,
        sphere_scale=2.0
    )
