from plyfile import PlyData, PlyElement
import numpy as np


"""
| 属性名        | 说明                           | 维度示例    |
| ---------- | ---------------------------- | ------- |
| `position` | 三维坐标 (x, y, z)               | (N, 3)  |
| `normal`   | 法向量 (nx, ny, nz)             | (N, 3)  |
| `f_dc`     | Spherical Harmonics 的常数项颜色   | (N, 3)  |
| `f_rest`   | Spherical Harmonics 其余项      | (N, 45) |
| `opacity`  | 每个高斯点的不透明度                   | (N,)    |
| `scale`    | 三轴缩放系数                       | (N, 3)  |
| `rotation` | 四元数旋转 (w, x, y, z 或 rot_0~3) | (N, 4)  |

"""
def load_gaussian_ply(ply_path: str):
    """
    读取高斯泼溅（Gaussian Splatting）格式的 PLY 文件
    返回一个结构化的字典：
    {
        'position': (N, 3),
        'normal': (N, 3),
        'f_dc': (N, 3),
        'f_rest': (N, M),
        'opacity': (N,),
        'scale': (N, 3),
        'rotation': (N, 4),
    }
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names

    # 提取常规字段
    pos = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    normal = np.stack([v["nx"], v["ny"], v["nz"]], axis=-1)

    # 提取 spherical harmonics (颜色系数)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1)

    # f_rest_0 ~ f_rest_44（45维） or f_rest_0 ~ f_rest_...（可变）
    f_rest_fields = [n for n in names if n.startswith("f_rest_")]
    f_rest = [v[name] for name in f_rest_fields]
    if len(f_rest) != 0:
        f_rest = np.stack(f_rest, axis=-1)
    else:
        f_rest = None

    # 透明度
    opacity = np.array(v["opacity"])

    # 三个缩放参数
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)

    # 四元数旋转
    rotation = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1)

    # 封装为结构化结果
    result = {
        "position": pos,
        "normal": normal,
        "f_dc": f_dc,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
    }
    if f_rest != None:
        result["f_rest"] = f_rest

    print(f"Loaded {len(pos)} Gaussian points from {ply_path}")
    return result


def save_gaussian_ply(result, save_path: str):
    """
    将高斯泼溅数据字典保存为 PLY 文件，格式与 Gaussian Splatting 原始输出一致
    """
    pos = result["position"]
    normal = result["normal"]
    f_dc = result["f_dc"]
    if "f_rest" in result:
        f_rest = result["f_rest"]
        n_rest = f_rest.shape[1]
    else:
        f_rest = None
        n_rest = 0
    opacity = result["opacity"]
    scale = result["scale"]
    rotation = result["rotation"]

    n_points = pos.shape[0]

    # 构造 PLY 文件字段描述
    vertex_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]

    # 动态扩展 f_rest 字段
    vertex_dtype += [(f"f_rest_{i}", "f4") for i in range(n_rest)]

    # 追加剩余属性
    vertex_dtype += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    # 初始化结构化数组
    vertex_all = np.empty(n_points, dtype=vertex_dtype)

    # 填充坐标和法向
    vertex_all["x"], vertex_all["y"], vertex_all["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    vertex_all["nx"], vertex_all["ny"], vertex_all["nz"] = normal[:, 0], normal[:, 1], normal[:, 2]

    # 填充颜色 DC
    vertex_all["f_dc_0"], vertex_all["f_dc_1"], vertex_all["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]

    # 填充高阶颜色系数 f_rest
    for i in range(n_rest):
        vertex_all[f"f_rest_{i}"] = f_rest[:, i]

    # 填充剩余属性
    vertex_all["opacity"] = opacity
    vertex_all["scale_0"], vertex_all["scale_1"], vertex_all["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
    vertex_all["rot_0"], vertex_all["rot_1"], vertex_all["rot_2"], vertex_all["rot_3"] = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

    # 封装为 PlyElement
    ply_element = PlyElement.describe(vertex_all, "vertex")

    # 写入二进制小端 PLY（与 COLMAP 和 Gaussian Splatting 兼容）
    PlyData([ply_element], text=False, byte_order="<").write(save_path)

    print(f"Saved {n_points} Gaussians to {save_path} (f_rest_dim={n_rest})")



if __name__ == "__main__":
    path = "your_file.ply"
    data = load_gaussian_ply(path)

    # 示例输出
    print("Positions:", data["position"].shape)
    print("Scales:", data["scale"].shape)
    print("Rotation:", data["rotation"].shape)
    print("f_rest:", data["f_rest"].shape)

