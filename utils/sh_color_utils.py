import numpy as np
import math

# 常数项 Y_00 = 1 / (2 * sqrt(pi))
Y00 = 0.28209479177

def rgb_to_fdc(rgb):
    """
    将RGB颜色转为球谐系数f_dc。
    输入:
        rgb: np.array([...])，可以是 (3,) 或 (N,3)
             支持 0-1 或 0-255 范围
    输出:
        f_dc: np.array([...])，与输入同形状
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    f_dc = rgb / Y00
    return f_dc


def fdc_to_rgb(f_dc):
    """
    将f_dc系数转回可视化RGB。
    输出范围为 [0, 1]
    """
    rgb = np.asarray(f_dc, dtype=np.float32) * Y00
    rgb = np.clip(rgb, 0, 1)
    return rgb



def rgb_to_sh0(rgb_color):
    """
    将RGB颜色转换为0阶球谐系数
    
    参数:
    rgb_color: RGB颜色值，格式为 [r, g, b]，取值范围0-1或0-255
    
    返回:
    sh_coeffs: 0阶球谐系数 [r_coeff, g_coeff, b_coeff]
    """
    # 确保RGB值在0-1范围内
    if isinstance(rgb_color, (list, tuple, np.ndarray)):
        rgb = np.array(rgb_color, dtype=np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    else:
        raise ValueError("RGB颜色应该是列表、元组或numpy数组")
    
    # 0阶球谐系数计算
    # 对于0阶，系数就是RGB值本身乘以常数因子
    # 0阶球谐基函数: Y00 = 0.5 * sqrt(1/pi)
    sh_coeffs = rgb * math.sqrt(4.0 * math.pi)
    
    return sh_coeffs

def sh0_to_rgb(sh_coeffs):
    """
    将0阶球谐系数转换回RGB颜色
    
    参数:
    sh_coeffs: 0阶球谐系数 [r_coeff, g_coeff, b_coeff]
    
    返回:
    rgb_color: RGB颜色 [r, g, b]，取值范围0-1
    """
    sh_coeffs = np.array(sh_coeffs, dtype=np.float32)
    
    # 反向转换
    rgb = sh_coeffs / math.sqrt(4.0 * math.pi)
    
    # 确保在0-1范围内
    rgb = np.clip(rgb, 0.0, 1.0)
    
    return rgb