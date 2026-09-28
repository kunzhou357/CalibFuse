"""calibfuse-metrics-v1 指标协议：15 个图像融合质量指标的 NumPy 实现。

**协议警告**：本评估器定义了固定的指标约定（包括若干与常见实现
不同的非常规约定，见下）。为保证历史与未来数值可比：
- 不许替换为其他实现、不许"就地修复"这些约定；
- 修正实现属于新协议（新版本号），不同协议的数值不可直接比较。

协议要点（跨代码库对比前必读）：
1. **指标在保存后的 8-bit PNG 上计算**——输出量化误差属于协议
   的一部分（见 utils/image.py 的 save_tensor）；
2. **MI 使用自然对数**（多数文献/代码用 log2），且先对每张图做
   min-max 归一化再量化到 256 级直方图；
3. **SSIM 与 VIF 对两个源求和**（vi→fused 与 ir→fused 相加），
   因此 SSIM 可以超过 1、VIF 大约为单源值的两倍量级；
4. **PSNR/RMSE 的"均方"实为 sqrt(SSE)/(m·n)**——不是标准的
   sqrt(SSE/(m·n))，系本协议的固定约定（_mse_quirk），数值显著
   小于常规 RMSE；
5. **Qabf 保留"梯度相等"分支**（ratio = g_f，常规实现多取 1）；
6. 灰度转换使用 BT.601 系数并做 MATLAB 式四舍五入
   （floor(x+0.5)，与 np.rint 的银行家舍入不同）。

指标一览（METRIC_ORDER 即 test.py 报表列序）：
EN 熵、CE 交叉熵、MI 互信息、PSNR、AG 平均梯度、EI 边缘强度、
Qabf 基于梯度的融合质量、SD 标准差、SF 空间频率、RMSE、SSIM、
Qcb/Qcv（Chen-Blum 与 Chen-Varma 感知协议）、SCD 相关系数和、
VIF 视觉信息保真度。
"""


from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image

# 报表列序；test.py 按此顺序输出 metrics.csv
METRIC_ORDER = ("EN", "CE", "MI", "PSNR", "AG", "EI", "Qabf", "SD", "SF", "RMSE", "SSIM", "Qcb", "Qcv", "SCD", "VIF")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _matlab_round(x: np.ndarray) -> np.ndarray:
    """MATLAB 式四舍五入：floor(x + 0.5)。

    与 numpy 的 rint（银行家舍入：0.5 取偶数）不同，历史 MATLAB
    实现如此；灰度转换与直方图量化均用它，差一个像素级会影响
    直方图类指标的可比性。
    """
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)

def _maybe_gray(img: np.ndarray) -> np.ndarray:
    """转灰度：二维原样返回；三维用 BT.601 系数加权并取整截断。

    系数是 MATLAB rgb2gray 的精确浮点值（非 0.299/0.587/0.114 的
    简化值），配合 _matlab_round 保持与历史实现比特一致。
    """
    if img.ndim == 2:
        return img
    coef = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])
    gray = img[..., 0] * coef[0] + img[..., 1] * coef[1] + img[..., 2] * coef[2]
    return np.clip(_matlab_round(gray), 0, 255)


def _conv2(x: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """FFT 实现的二维卷积（MATLAB conv2 语义：核翻转的真卷积）。

    用 FFT 而非 scipy.signal 是为了不引入额外依赖；对 3x3 核与
    小图足够快且数值精度可用 float64 保证。

    参数:
        x: 输入图 (H, W)。
        kernel: 卷积核 (kh, kw)。
        mode: ``"same"`` 输出同尺寸（居中裁剪）；``"valid"`` 只输出
            完全覆盖区域。

    返回:
        卷积结果（float64）。
    """
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    # 零填充到"full"尺寸后在频域相乘
    fh, fw = x.shape[0] + kh - 1, x.shape[1] + kw - 1
    full = np.fft.irfft2(
        np.fft.rfft2(x, (fh, fw)) * np.fft.rfft2(kernel, (fh, fw)), (fh, fw)
    )
    if mode == "valid":
        return full[kh - 1 : full.shape[0] - kh + 1, kw - 1 : full.shape[1] - kw + 1]
    if mode == "same":
        return full[(kh - 1) // 2 : (kh - 1) // 2 + x.shape[0], (kw - 1) // 2 : (kw - 1) // 2 + x.shape[1]]
    raise ValueError(f"unsupported convolution mode: {mode}")


def _filter2_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """MATLAB filter2('same') 语义：相关（不翻转核）。"""
    return _conv2(x, np.asarray(kernel)[::-1, ::-1], "same")

def _filter2_valid(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """filter2('valid') 语义的相关运算（不翻转核，valid 裁剪）。"""
    return _conv2(x, np.asarray(kernel)[::-1, ::-1], "valid")

def _imfilter_replicate(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """边缘复制填充 + valid 相关（MATLAB imfilter 默认 'replicate'）。"""
    kh, kw = kernel.shape
    xp = np.pad(np.asarray(x, dtype=np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
    return _filter2_valid(xp, kernel)

def _freqspace_1d(n: int) -> np.ndarray:
    """MATLAB freqspace 的 1D 归一化频率轴 [-1, 1)。

    偶数：[-n/2, n/2)/(n/2)；奇数：[-d, d]/d。供 CSF 频率网格用。
    """
    if n % 2 == 0:
        return np.arange(-(n // 2), n // 2) / (n / 2.0)
    d = (n - 1) // 2
    return np.arange(-d, d + 1) / float(d)

def _freqspace_meshgrid(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """二维频率网格 (u 列向, v 行向)，值域 [-1,1)。"""
    f_rows, f_cols = _freqspace_1d(rows), _freqspace_1d(cols)
    return np.meshgrid(f_cols, f_rows)

def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    """归一化的方形高斯窗（SSIM 的 11x11、σ=1.5 标准窗）。"""
    c = (size - 1) / 2.0
    y, x = np.ogrid[-c : c + 1, -c : c + 1]
    w = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return w / w.sum()


def _gaussian2d_31(sigma: float) -> np.ndarray:
    """31x31 未归一化的二维高斯（Qcb 的 DoG 对比度用）。"""
    y, x = np.mgrid[-15:16, -15:16]
    return np.exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / (2.0 * math.pi * sigma * sigma)


def read_image(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """读取图像为 float64 数组（H,W）或 (H,W,3)。

    调色板图转 RGB；RGBA 截取前三通道；需要时可双线性缩放到
    指定 (rows, cols)（先量化回 uint8 再缩放，保持 8-bit 语义）。

    参数:
        path: 图像路径。
        size: 可选的目标 (rows, cols)；与当前不同才缩放。

    返回:
        float64 数组，取值 [0,255]。
    """
    with Image.open(path) as im:
        if im.mode == "P":
            im = im.convert("RGB")
        arr = np.asarray(im)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    arr = arr.astype(np.float64)
    if size is not None and arr.shape[:2] != tuple(size):
        # 与源图对齐（尺寸不一致的红外等）：回 uint8 再双线性缩放
        u8 = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
        arr = np.asarray(u8.resize((size[1], size[0]), Image.BILINEAR)).astype(np.float64)
    return arr


def _normalize255(img: np.ndarray) -> np.ndarray:
    """min-max 拉伸到 [0,255] 并 MATLAB 取整；全零图原样返回。"""
    lo, hi = float(img.min()), float(img.max())
    if hi == 0.0 and lo == 0.0:
        return img.astype(np.float64)
    return _matlab_round((img - lo) / (hi - lo) * 255.0)


def _hist256(img: np.ndarray) -> np.ndarray:
    """256 级灰度直方图（归一化为概率）。"""
    values = np.clip(np.rint(img), 0, 255).astype(np.int64).ravel()
    return np.bincount(values, minlength=256).astype(np.float64) / values.size


def _blkproc_sum_pow(x: np.ndarray, block: int, power: float) -> np.ndarray:
    """按 block×block 分块求幂和（Qcv 的局部显著度用）。

    不足一块的右/下边缘零填充。
    """
    rows, cols = x.shape
    pr, pc = (-rows) % block, (-cols) % block
    if pr or pc:
        x = np.pad(x, ((0, pr), (0, pc)))
    nb_r, nb_c = x.shape[0] // block, x.shape[1] // block
    blocks = x.reshape(nb_r, block, nb_c, block)
    return np.sum(blocks ** power, axis=(1, 3))


def _blkproc_mean_square(x: np.ndarray, block: int) -> np.ndarray:
    """按 block×block 分块求均方值（Qcv 的局部失真用）。"""
    rows, cols = x.shape
    pr, pc = (-rows) % block, (-cols) % block
    if pr or pc:
        x = np.pad(x, ((0, pr), (0, pc)))
    nb_r, nb_c = x.shape[0] // block, x.shape[1] // block
    blocks = x.reshape(nb_r, block, nb_c, block)
    return np.mean(blocks * blocks, axis=(1, 3))


def _vifb_metric(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray, single) -> float:
    """单通道指标的多通道适配器。

    约定：**红外恒为单通道**；融合图为灰度时直接调用；为 RGB 时
    逐通道调用后取平均。各 *_single 指标经由本函数包装成对外接口。

    参数:
        vi, ir, fused: 输入图（vi 可为 3 通道，ir 单通道）。
        single: 单通道指标函数。

    返回:
        标量指标值。
    """
    b = 1 if fused.ndim == 2 else fused.shape[2]
    b1 = 1 if ir.ndim == 2 else ir.shape[2]
    if b == 1:
        return single(vi, ir, fused)
    if b1 == 1:
        vals = [single(vi[..., k], ir, fused[..., k]) for k in range(b)]
    else:
        vals = [single(vi[..., k], ir[..., k], fused[..., k]) for k in range(b)]
    return float(np.mean(vals))

def _en_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """EN 熵：融合图灰度直方图的 Shannon 熵（bit）。越高信息量越大。"""
    p = _hist256(fused)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def _ce_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """CE 交叉熵：源图与融合图交叉熵的平均（越低越好）。"""
    def ce_pair(a: np.ndarray, f: np.ndarray) -> float:
        p1, p2 = _hist256(_maybe_gray(a)), _hist256(_maybe_gray(f))
        mask = (p1 > 0) & (p2 > 0)
        return float(np.sum(p1[mask] * np.log2(p1[mask] / p2[mask])))

    return (ce_pair(vi, fused) + ce_pair(ir, fused)) / 2.0

def _mi_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """MI 互信息（**协议怪癖：自然对数**）。

    每张图先 min-max 归一化并量化到 256 级，再从 256×256 联合
    直方图估计互信息；结果为 vi-fused 与 ir-fused 两项之和。
    注意用的是 ln 而非 log2——与多数文献实现不同，系历史实现
    遗留，勿改。
    """
    def mi_pair(a: np.ndarray, b: np.ndarray) -> float:
        def norm(x: np.ndarray) -> np.ndarray:
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo) if hi != lo else np.zeros_like(x)
        ai = np.clip(_matlab_round(norm(_maybe_gray(a)) * 255.0), 0, 255).astype(np.int64).ravel()
        bi = np.clip(_matlab_round(norm(_maybe_gray(b)) * 255.0), 0, 255).astype(np.int64).ravel()
        # 联合直方图：ai*256+bi 编码到一维再还原
        joint = np.bincount(ai * 256 + bi, minlength=256 * 256).reshape(256, 256).astype(np.float64)
        p_joint = joint / joint.sum()
        ha = np.bincount(ai, minlength=256).astype(np.float64); ha /= ha.sum()
        hb = np.bincount(bi, minlength=256).astype(np.float64); hb /= hb.sum()
        hab = p_joint[p_joint > 0]
        ha_n, hb_n = ha[ha > 0], hb[hb > 0]
        # I(A;B) = H(A) + H(B) - H(A,B)，全部用自然对数
        return float(-np.sum(ha_n * np.log(ha_n)) - np.sum(hb_n * np.log(hb_n)) + np.sum(hab * np.log(hab)))
    return mi_pair(vi, fused) + mi_pair(ir, fused)

def _mse_quirk(a: np.ndarray, b: np.ndarray) -> float:
    """本协议的"伪 MSE"：sqrt(SSE) / (m·n)。

    **协议怪癖**：常规 RMSE 是 sqrt(SSE/(m·n))，这里多除了一次
    (m·n)。数值因此远小于常规 RMSE，PSNR 也随之偏高。此约定是
    calibfuse-metrics-v1 协议的一部分，本项目所有数值都基于它，勿改。
    """
    m, n = a.shape
    return float(math.sqrt(np.sum((a - b) ** 2)) / (m * n))


def _psnr_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """PSNR：两个源方向的 _mse_quirk 取平均后代入 20·log10(255/rmse)。

    完全相同时返回 inf。
    """
    mes = (_mse_quirk(_maybe_gray(vi), _maybe_gray(fused)) + _mse_quirk(_maybe_gray(ir), _maybe_gray(fused))) / 2.0
    return float("inf") if mes == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mes)))


def _rmse_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """RMSE：两个源方向的 _mse_quirk 平均（见 _mse_quirk 的怪癖说明）。"""
    return (_mse_quirk(_maybe_gray(vi), _maybe_gray(fused)) + _mse_quirk(_maybe_gray(ir), _maybe_gray(fused))) / 2.0


def _ag_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """AG 平均梯度：梯度幅值（dx²+dy²)/2 开方）总和 / ((H-1)(W-1))。"""
    dy, dx = np.gradient(fused)
    s = np.sqrt((dx * dx + dy * dy) / 2.0)
    return float(np.sum(s) / ((fused.shape[0] - 1) * (fused.shape[1] - 1)))


def _ei_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """EI 边缘强度：Sobel 梯度幅值的均值。"""
    w = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    gx = _imfilter_replicate(fused, w)
    gy = _imfilter_replicate(fused, w.T)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _qabf_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """Qabf：基于边缘强度与方向的融合质量（Xydeas-Živković）。

    对每个源计算 Sobel 梯度 g 与角度 a，用 sigmoid 形的 qg/qa
    评价融合图对源边缘"强度比"与"方向对齐"的保持度，再按源梯度
    加权平均。

    **协议怪癖**：保留了"梯度相等"分支 ``ratio = g_f``
    （常规实现多取 1）；以及 |gx|<1e-6 直接置 0（影响角度计算
    的零点处理）。
    """
    # 两个 Sobel 核（h1=纵向差分，h3=横向差分）
    h1 = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    h3 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    # sigmoid 参数：强度与方向的评价曲线拐点/斜率（历史常用值）
    tg, kg, dg = 0.9994, -15.0, 0.5
    ta, ka, da = 0.9879, -22.0, 0.8

    def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = _conv2(_maybe_gray(img), h3, "same")
        gy = _conv2(_maybe_gray(img), h1, "same")
        gx[np.abs(gx) < 1e-6] = 0.0
        gy[np.abs(gy) < 1e-6] = 0.0
        g = np.sqrt(gx * gx + gy * gy)
        # 默认角度 π/2（gx=0 时），否则 arctan(gy/gx)
        ang = np.full(gx.shape, math.pi / 2.0)
        nz = gx != 0
        ang[nz] = np.arctan(gy[nz] / gx[nz])
        return g, ang

    def quality(g_s: np.ndarray, a_s: np.ndarray, g_f: np.ndarray, a_f: np.ndarray) -> np.ndarray:
        # 强度比：g_s>g_f 时取 g_f/g_s，反之取倒数——衡量幅值保持
        ratio = np.zeros_like(g_s)
        gt, eq, lt = g_s > g_f, g_s == g_f, g_s < g_f
        ratio[gt] = g_f[gt] / g_s[gt]
        ratio[eq] = g_f[eq]  # 历史分支：相等时直接用 g_f（而非 1）
        ratio[lt] = g_s[lt] / g_f[lt]
        # 方向对齐度：角度差归一到 [0,1]
        align = 1.0 - np.abs(a_s - a_f) / (math.pi / 2.0)
        qg = tg / (1.0 + np.exp(kg * (ratio - dg)))
        qa = ta / (1.0 + np.exp(ka * (align - da)))
        return qg * qa

    ga, aa = sobel(vi)
    gb, ab = sobel(ir)
    gf, af = sobel(fused)
    qa = quality(ga, aa, gf, af)
    qb = quality(gb, ab, gf, af)
    # 按源梯度强度加权平均两个方向的质量
    return float(np.sum(qa * ga + qb * gb) / np.sum(ga + gb))


def _scd_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """SCD：源与"差分互补图"的相关系数之和（Haghighat-Resetti）。

    corr(a, f - b) 衡量融合图相对另一源新增的信息与源 a 的相关性，
    两个方向求和；满分约 2。
    """
    def corr(x: np.ndarray, y: np.ndarray) -> float:
        x, y = x - x.mean(), y - y.mean()
        denominator = math.sqrt(np.sum(x * x) * np.sum(y * y))
        return float(np.sum(x * y) / denominator) if denominator else 0.0

    a, b, f = _maybe_gray(vi), _maybe_gray(ir), _maybe_gray(fused)
    return corr(a, f - b) + corr(b, f - a)


def _sd_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """SD 标准差：融合图灰度总体标准差，衡量对比度。"""
    return float(np.sqrt(np.sum((fused - fused.mean()) ** 2) / fused.size))


def _sf_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """SF 空间频率：行/列差分能量（各除以 m·n）开方。"""
    m, n = fused.shape
    rf = np.sum((fused[:, 1:] - fused[:, :-1]) ** 2) / (m * n)
    cf = np.sum((fused[1:, :] - fused[:-1, :]) ** 2) / (m * n)
    return float(math.sqrt(rf + cf))


def _ssim_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """SSIM（**协议怪癖：两源求和，可超过 1**）。

    标准 11x11 高斯窗（σ=1.5）、C1/C2 取 (0.01·255)²/(0.03·255)²，
    逐窗计算后整图平均；结果是 ssim(vi,f) + ssim(ir,f)。
    """
    def ssim_pair(a: np.ndarray, b: np.ndarray) -> float:
        w = _gaussian_window(11, 1.5)
        c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
        a, b = _maybe_gray(a), _maybe_gray(b)
        # 高斯窗加权局部统计（相关实现）
        ua, ub = _filter2_valid(a, w), _filter2_valid(b, w)
        ua_ub = ua * ub
        sig_a = _filter2_valid(a * a, w) - ua * ua
        sig_b = _filter2_valid(b * b, w) - ub * ub
        sig_ab = _filter2_valid(a * b, w) - ua_ub
        ssim_map = ((2.0 * ua_ub + c1) * (2.0 * sig_ab + c2)) / (
            (ua * ua + ub * ub + c1) * (sig_a + sig_b + c2)
        )
        return float(ssim_map.mean())

    return ssim_pair(vi, fused) + ssim_pair(ir, fused)


def _qcb_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """Qcb（Chen-Blum 2009 感知协议）。

    流程：CSF 频域滤波（DoG 形对比敏感度函数）→ 局部对比度
    （两个 σ 的高斯 DoG）→ 对比度 pooling（幂律压缩）→ 源显著度
    加权平均两个方向的保持度 q。
    """
    im1, im2, imf = _normalize255(_maybe_gray(vi)), _normalize255(_maybe_gray(ir)), _normalize255(_maybe_gray(fused))

    # CSF 参数（Chen-Blum 论文值）：f0/f1 频点、a_csf 负瓣深度
    f0, f1, a_csf = 15.3870, 1.3456, 0.7622
    k, h_c, p, q, z = 1.0, 1.0, 3.0, 2.0, 0.0001
    rows, cols = im1.shape
    # 归一化频率网格的纵横尺度
    hh, ll = rows / 30.0, cols / 30.0
    u, v = _freqspace_meshgrid(rows, cols)
    r = np.sqrt((ll * u) ** 2 + (hh * v) ** 2)
    # CSF = 两个高斯的差（band-pass）
    sd = np.exp(-(r / f0) ** 2) - a_csf * np.exp(-(r / f1) ** 2)

    def csf_filter(im: np.ndarray) -> np.ndarray:
        spec = np.fft.fftshift(np.fft.fft2(im)) * sd
        return np.fft.ifft2(np.fft.ifftshift(spec)).real

    # 两个尺度的 31x31 高斯（σ=2 / σ=4），构成 DoG 对比度探测
    g1, g2 = _gaussian2d_31(2.0), _gaussian2d_31(4.0)

    def contrast(im: np.ndarray) -> np.ndarray:
        return _filter2_same(im, g1) / _filter2_same(im, g2) - 1.0

    def c_pooled(im: np.ndarray) -> np.ndarray:
        c = np.abs(contrast(csf_filter(im)))
        return (k * c ** p) / (h_c * c ** q + z)

    # 除零容忍：0/0 产生 NaN 的位置在掩码选择后不进入均值（下同）
    with np.errstate(divide="ignore", invalid="ignore"):
        c1p, c2p, cfp = c_pooled(im1), c_pooled(im2), c_pooled(imf)
        # 每个方向取较小/较大比值（保持度 ∈ (0,1]）
        mask = c1p < cfp
        q1f = (c1p / cfp) * mask + (cfp / c1p) * (~mask)
        mask = c2p < cfp
        q2f = (c2p / cfp) * mask + (cfp / c2p) * (~mask)

    # 显著度权重：各源 pooled 对比度的归一化占比
    ramda1 = (c1p * c1p) / (c1p * c1p + c2p * c2p)
    ramda2 = (c2p * c2p) / (c1p * c1p + c2p * c2p)
    return float(np.mean(ramda1 * q1f + ramda2 * q2f))


def _qcv_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """Qcv（Chen-Varma 加权失真协议，越低越好）。

    以各源梯度显著度（16x16 块、5 次幂）为权重，加权平均融合图
    与每个源经 CSF 滤波后的块均方差。
    """
    im1, im2, imf = _normalize255(_maybe_gray(vi)), _normalize255(_maybe_gray(ir)), _normalize255(_maybe_gray(fused))

    flt1 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    flt2 = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])

    def grad_mag(im: np.ndarray) -> np.ndarray:
        gx, gy = _filter2_same(im, flt1), _filter2_same(im, flt2)
        return np.sqrt(gx * gx + gy * gy)

    # 分块窗口 16、显著度幂 5
    window, alpha = 16, 5
    ramda1 = _blkproc_sum_pow(grad_mag(im1), window, alpha)
    ramda2 = _blkproc_sum_pow(grad_mag(im2), window, alpha)

    rows, cols = im1.shape
    u, v = _freqspace_meshgrid(rows, cols)
    r = np.sqrt((cols / 8.0 * u) ** 2 + (rows / 8.0 * v) ** 2)

    # 人眼调制传递函数（Chen-Varma 论文式）
    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-((0.144 * r) ** 1.1))

    def csf_filter(diff: np.ndarray) -> np.ndarray:
        spec = np.fft.fftshift(np.fft.fft2(diff)) * theta_m
        return np.fft.ifft2(np.fft.ifftshift(spec)).real

    # 失真 = 源与融合图之差经 CSF 后的分块均方
    d1 = _blkproc_mean_square(csf_filter(im1 - imf), window)
    d2 = _blkproc_mean_square(csf_filter(im2 - imf), window)
    return float(np.sum(ramda1 * d1 + ramda2 * d2) / np.sum(ramda1 + ramda2))


def _vif_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """VIF 视觉信息保真度（Sheikh-Bovik，**协议怪癖：两源求和**）。

    标准 4 尺度实现：每尺度用高斯窗估计局部均值/方差/协方差，
    计算 g（增益）与 sv_sq（残差噪声），累加 log10 的信息比；
    尺度间对图 2 倍降采样。结果 = vif(vi→fused) + vif(ir→fused)。
    """
    def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        radius = (size - 1) / 2.0
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        # 过小的权重截断为 0（MATLAB fspecial 同款做法）
        kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
        total = kernel.sum()
        return kernel / total if total != 0 else kernel

    def compare_vif(reference: np.ndarray, distorted: np.ndarray) -> float:
        reference = _maybe_gray(reference).astype(np.float64)
        distorted = _maybe_gray(distorted).astype(np.float64)
        sigma_nsq, eps = 2, 1e-10
        num = den = 0.0
        ref, dist = reference.copy(), distorted.copy()
        for scale in range(1, 5):
            # 尺度越大窗口越小；σ = N/5（标准 VIF 设置）
            N = 2 ** (5 - scale) + 1
            win = np.rot90(gaussian_kernel(N, N / 5.0), 2)
            if scale > 1:
                # 非首个尺度：先 valid 低通再 2 倍抽取
                ref = _conv2(ref, win, "valid")[::2, ::2]
                dist = _conv2(dist, win, "valid")[::2, ::2]
            # 局部统计（相关实现）
            mu1 = _conv2(ref, win, "valid")
            mu2 = _conv2(dist, win, "valid")
            sigma1_sq = _conv2(ref * ref, win, "valid") - mu1 * mu1
            sigma2_sq = _conv2(dist * dist, win, "valid") - mu2 * mu2
            sigma12 = _conv2(ref * dist, win, "valid") - mu1 * mu2
            # 数值防御：负方差截零
            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0
            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            # 退化区域掩码处理（标准 VIF 的分支逻辑）
            mask = sigma1_sq < eps
            g[mask], sv_sq[mask], sigma1_sq[mask] = 0, sigma2_sq[mask], 0
            mask = sigma2_sq < eps
            g[mask], sv_sq[mask] = 0, 0
            mask = g < 0
            sv_sq[mask], g[mask] = sigma2_sq[mask], 0
            sv_sq[sv_sq <= eps] = eps
            num += np.sum(np.log10(1 + (g * g * sigma1_sq) / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
        if den == 0:
            return 1.0
        vif = num / den
        return float(vif) if not np.isnan(vif) else 1.0

    return compare_vif(vi, fused) + compare_vif(ir, fused)


def _make_metric(single):
    """把单通道指标包装成支持 RGB 融合图的多通道入口。"""
    def metric(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
        return _vifb_metric(vi, ir, fused, single)
    return metric


# —— 对外指标接口（RGB 融合图自动逐通道适配）——
EN = _make_metric(_en_single)
CE = _make_metric(_ce_single)
MI = _make_metric(_mi_single)
PSNR = _make_metric(_psnr_single)
AG = _make_metric(_ag_single)
EI = _make_metric(_ei_single)
Qabf = _make_metric(_qabf_single)
SCD = _make_metric(_scd_single)
SD = _make_metric(_sd_single)
SF = _make_metric(_sf_single)
RMSE = _make_metric(_rmse_single)
SSIM = _make_metric(_ssim_single)
Qcb = _make_metric(_qcb_single)
Qcv = _make_metric(_qcv_single)
VIF = _make_metric(_vif_single)

METRIC_FUNCS = {
    "EN": EN, "CE": CE, "MI": MI, "PSNR": PSNR, "AG": AG, "EI": EI,
    "Qabf": Qabf, "SD": SD, "SF": SF, "RMSE": RMSE, "SSIM": SSIM,
    "Qcb": Qcb, "Qcv": Qcv, "SCD": SCD, "VIF": VIF,
}

def evaluate(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray, metrics: tuple[str, ...]) -> dict[str, float]:
    """计算指定的指标集合。

    参数:
        vi: 可见光源图（灰度或 RGB，float64 0~255）。
        ir: 红外源图（单通道）。
        fused: 融合结果（灰度或 RGB）。
        metrics: 指标名元组（须在 METRIC_FUNCS 中）。

    返回:
        ``{指标名: 数值}`` 字典。
    """
    return {name: METRIC_FUNCS[name](vi, ir, fused) for name in metrics}
