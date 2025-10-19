import numpy as np

# 常数 Y00 = 1 / (2 * sqrt(pi))
Y00 = 0.28209479177387814

def srgb_to_linear(rgb):
    """
    把 sRGB (0-1 或 0-255) 转为线性光空间。
    支持形状 (..., 3)。
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size == 0:
        return rgb
    # 规范到 0..1
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    # sRGB -> linear
    a = 0.055
    linear = np.where(rgb <= 0.04045,
                      rgb / 12.92,
                      ((rgb + a) / (1.0 + a)) ** 2.4)
    return linear


def linear_to_srgb(linear):
    """
    线性光 -> sRGB (0..1)，支持形状 (...,3)。
    """
    linear = np.asarray(linear, dtype=np.float32)
    if linear.size == 0:
        return linear
    a = 0.055
    srgb = np.where(linear <= 0.0031308,
                    linear * 12.92,
                    (1.0 + a) * (linear ** (1.0 / 2.4)) - a)
    return np.clip(srgb, 0.0, 1.0)


def rgb_to_fdc(rgb):
    """
    将 sRGB RGB -> f_dc（SH 的 DC 分量 c00）。
    输入 rgb 支持 (...,3)，值可在 0-1 或 0-255。
    返回 f_dc 与输入同形状，单位是线性亮度 / Y00（可以大于1）。
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    # 保证最后一维是 3
    if rgb.ndim == 1 and rgb.shape[0] == 3:
        squeezed = True
    else:
        squeezed = False
    lin = srgb_to_linear(rgb)
    f_dc = lin / Y00
    return f_dc[0] if squeezed else f_dc


def fdc_to_rgb(f_dc):
    """
    将 f_dc（SH DC 分量）转回可视化 sRGB（范围 0..1）。
    输入 f_dc 支持 (...,3)，返回与输入同形状。
    """
    f_dc = np.asarray(f_dc, dtype=np.float32)
    lin = f_dc * Y00
    srgb = linear_to_srgb(lin)
    return srgb
