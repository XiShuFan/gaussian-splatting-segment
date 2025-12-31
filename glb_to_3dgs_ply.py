import trimesh
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from trimesh.visual import ColorVisuals, TextureVisuals
from sklearn.neighbors import NearestNeighbors

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

# =========================
# 参数区
# =========================
PROJECT = "aniu"
GLB_PATH = f"/media/why/新加卷/xsf/商品3DGS/mesh_to_gs/{PROJECT}/{PROJECT}.glb"
OUTPUT_PLY = f"/media/why/新加卷/xsf/商品3DGS/mesh_to_gs/{PROJECT}/{PROJECT}_3dgs.ply"

# 采样点数量
NUM_POINTS = 300_000
# 不透明度
ALPHA = 0.9999
# 缩放系数
SCALE_FACTOR = 3.0


# =========================
# 工具函数
# =========================
def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


# =========================
# 1. 读取 GLB
# =========================
mesh = trimesh.load(GLB_PATH, force='mesh')

assert mesh.is_watertight or len(mesh.faces) > 0
print(f"[INFO] Mesh loaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

# =========================
# 2. 表面采样
# =========================
points, face_idx = trimesh.sample.sample_surface(mesh, NUM_POINTS)

# =========================
# 3. 计算 barycentric 坐标（用于 UV 插值）
# =========================
triangles = mesh.triangles[face_idx]
bary = trimesh.triangles.points_to_barycentric(triangles, points)

# =========================
# 4. 颜色采样（支持 UV + 贴图）
# =========================
def get_basecolor_image(material):
    """
    Return baseColor image as numpy array (H,W,3) or None
    Compatible with different trimesh glTF behaviors.
    """
    if material is None:
        return None

    tex = getattr(material, "baseColorTexture", None)
    if tex is None:
        return None

    # 情况 1：已经是 PIL Image
    if hasattr(tex, "size") and hasattr(tex, "mode"):
        img = np.array(tex)

    # 情况 2：trimesh Texture 对象
    elif hasattr(tex, "image"):
        img = np.array(tex.image)

    # 情况 3：已经是 numpy
    elif isinstance(tex, np.ndarray):
        img = tex

    else:
        return None

    # RGBA → RGB
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    return img.astype(np.float32) / 255.0



if isinstance(mesh.visual, TextureVisuals):
    print("[INFO] Using PBR baseColorTexture")

    uvs = mesh.visual.uv
    faces = mesh.faces

    uv_tri = uvs[faces[face_idx]]
    uv = (uv_tri * bary[..., None]).sum(axis=1)

    tex_image = get_basecolor_image(mesh.visual.material)

    if tex_image is None:
        raise RuntimeError("Failed to retrieve baseColorTexture")

    h, w = tex_image.shape[:2]

    u = np.clip((uv[:, 0] * w).astype(np.int32), 0, w - 1)
    v = np.clip(((1.0 - uv[:, 1]) * h).astype(np.int32), 0, h - 1)

    colors = tex_image[v, u]

elif isinstance(mesh.visual, ColorVisuals):
    print("[INFO] Using vertex colors")

    vc = mesh.visual.vertex_colors[:, :3] / 255.0
    face_vc = vc[mesh.faces[face_idx]]
    colors = (face_vc * bary[..., None]).sum(axis=1)

else:
    print("[WARN] No color info found, using gray")
    colors = np.full((NUM_POINTS, 3), 0.7)


# =========================
# 5. 法线（用于 rotation）
# =========================
normals = mesh.face_normals[face_idx]
normals = normalize(normals)

# =========================
# 6. Scale 估计（关键）
# =========================
points_np = np.asarray(points, dtype=np.float32)  # (N,3)
# 找每个点的最近邻（k=2，第 0 个是自己）
nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(points_np)
distances, indices = nbrs.kneighbors(points_np)
# distances[:, 0] = 0（自己）
nearest_dist = distances[:, 1]        # 最近邻距离
dist2 = np.clip(nearest_dist ** 2, 1e-7, None) * SCALE_FACTOR
# 对应 3DGS 的 log(sqrt(dist2))
scale_scalar = np.log(np.sqrt(dist2))
# 各向同性 scale
# scales = np.repeat(scale_scalar[:, None], 3, axis=1)
# 各向异性 scale
normal  = scale_scalar - 1.0
scales = np.stack([scale_scalar, scale_scalar, normal], axis=1)

# =========================
# 7. Rotation（简化：identity）
# Super Splat / 多数 GS Viewer 对 rotation 不敏感
# =========================
rotations = np.tile([0.0, 0.0, 0.0, 0.0], (NUM_POINTS, 1))  # quaternion (w,x,y,z)

# =========================
# 8. Alpha
# =========================
alpha = np.full((NUM_POINTS, 1), ALPHA)

# =========================
# 9. 写出 3DGS PLY
# =========================
NUM_REST = 24  # f_rest_0 ~ f_rest_23

vertex_dtype = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
]

# 补 f_rest_0 ~ f_rest_23
# for i in range(NUM_REST):
#     vertex_dtype.append((f"f_rest_{i}", "f4"))

# 其余高斯参数
vertex_dtype.extend([
    ("opacity", "f4"),
    ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
    ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
])

vertex_data = np.empty(NUM_POINTS, dtype=vertex_dtype)


"""
高斯参数
ply
format binary_little_endian 1.0
element vertex 233622
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
property float f_rest_1
property float f_rest_2
property float f_rest_3
property float f_rest_4
property float f_rest_5
property float f_rest_6
property float f_rest_7
property float f_rest_8
property float f_rest_9
property float f_rest_10
property float f_rest_11
property float f_rest_12
property float f_rest_13
property float f_rest_14
property float f_rest_15
property float f_rest_16
property float f_rest_17
property float f_rest_18
property float f_rest_19
property float f_rest_20
property float f_rest_21
property float f_rest_22
property float f_rest_23
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

"""
普通参数
ply
format ascii 1.0
comment VCGLIB generated
element vertex 8348
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face 16000
property list uchar int vertex_indices
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
"""

# 高斯中心坐标
"""
注意，高斯展示的正确方向是
vertex_data["x"] = -points[:, 0]
vertex_data["y"] = -points[:, 1]
"""
vertex_data["x"] = points[:, 0]
vertex_data["y"] = points[:, 1]
vertex_data["z"] = points[:, 2]

# 球谐函数颜色
fused_color = RGB2SH(colors)
vertex_data["f_dc_0"] = fused_color[:, 0]
vertex_data["f_dc_1"] = fused_color[:, 1]
vertex_data["f_dc_2"] = fused_color[:, 2]

# for i in range(NUM_REST):
#     vertex_data[f"f_rest_{i}"] = 0.0

# 透明度
def inverse_sigmoid(x):
    return np.log(x/(1-x))
vertex_data["opacity"] = inverse_sigmoid(alpha[:, 0])

# 缩放
vertex_data["scale_0"] = scales[:, 0]
vertex_data["scale_1"] = scales[:, 1]
vertex_data["scale_2"] = scales[:, 2]

vertex_data["rot_0"] = rotations[:, 0]
vertex_data["rot_1"] = rotations[:, 1]
vertex_data["rot_2"] = rotations[:, 2]
vertex_data["rot_3"] = rotations[:, 3]

# 是否输出二进制 text
ply = PlyData([PlyElement.describe(vertex_data, "vertex")], text=False)
ply.write(OUTPUT_PLY)

print(f"[DONE] 3DGS PLY written to: {OUTPUT_PLY}")
