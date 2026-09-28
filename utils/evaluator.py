"""calibfuse-metrics-v1 metric protocol: NumPy implementations of 15 fusion metrics.

Protocol warning: this evaluator fixes several unconventional conventions.
For comparability, do not replace it or "fix" these conventions in place —
a corrected implementation is a new protocol, and values from different
protocols must not be compared.

Key conventions:
1. Metrics are computed on the saved 8-bit PNGs.
2. MI uses natural logs (not log2) on 256-level histograms.
3. SSIM and VIF sum the two source scores (SSIM can exceed 1).
4. The PSNR/RMSE "mean square" is ``sqrt(SSE) / (m * n)``, not
   ``sqrt(SSE / (m * n))``.
5. Qabf keeps the equal-gradient branch (``ratio = g_f``).
6. Grayscale conversion uses the exact MATLAB ``rgb2gray`` coefficients
   with MATLAB rounding (``floor(x + 0.5)``).

Metrics (METRIC_ORDER is the report column order): EN entropy, CE cross
entropy, MI mutual information, PSNR, AG average gradient, EI edge
intensity, Qabf gradient-based fusion quality, SD standard deviation, SF
spatial frequency, RMSE, SSIM, Qcb/Qcv (Chen-Blum and Chen-Varma perceptual
protocols), SCD sum of correlations, VIF visual information fidelity.
"""


from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image

# Report column order used by test.py for metrics.csv.
METRIC_ORDER = ("EN", "CE", "MI", "PSNR", "AG", "EI", "Qabf", "SD", "SF", "RMSE", "SSIM", "Qcb", "Qcv", "SCD", "VIF")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _matlab_round(x: np.ndarray) -> np.ndarray:
    """MATLAB rounding: ``floor(x + 0.5)`` (differs from np.rint at .5)."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)

def _maybe_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale conversion with the exact MATLAB rgb2gray coefficients."""
    if img.ndim == 2:
        return img
    coef = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])
    gray = img[..., 0] * coef[0] + img[..., 1] * coef[1] + img[..., 2] * coef[2]
    return np.clip(_matlab_round(gray), 0, 255)


def _conv2(x: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """FFT-based 2D convolution with MATLAB ``conv2`` semantics (kernel flipped)."""
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    # multiply in the frequency domain on the zero-padded "full" size
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
    """MATLAB ``filter2('same')`` semantics: correlation (kernel not flipped)."""
    return _conv2(x, np.asarray(kernel)[::-1, ::-1], "same")

def _filter2_valid(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """MATLAB ``filter2('valid')`` semantics: correlation, valid cropping."""
    return _conv2(x, np.asarray(kernel)[::-1, ::-1], "valid")

def _imfilter_replicate(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Edge-replicate padding + valid correlation (MATLAB ``imfilter`` default)."""
    kh, kw = kernel.shape
    xp = np.pad(np.asarray(x, dtype=np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
    return _filter2_valid(xp, kernel)

def _freqspace_1d(n: int) -> np.ndarray:
    """MATLAB ``freqspace`` 1D normalized frequency axis in [-1, 1)."""
    if n % 2 == 0:
        return np.arange(-(n // 2), n // 2) / (n / 2.0)
    d = (n - 1) // 2
    return np.arange(-d, d + 1) / float(d)

def _freqspace_meshgrid(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """2D frequency grid (u across columns, v down rows) in [-1, 1)."""
    f_rows, f_cols = _freqspace_1d(rows), _freqspace_1d(cols)
    return np.meshgrid(f_cols, f_rows)

def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    """Normalized square Gaussian window (SSIM default: 11x11, sigma 1.5)."""
    c = (size - 1) / 2.0
    y, x = np.ogrid[-c : c + 1, -c : c + 1]
    w = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return w / w.sum()


def _gaussian2d_31(sigma: float) -> np.ndarray:
    """Unnormalized 31x31 2D Gaussian (Qcb DoG contrast detection)."""
    y, x = np.mgrid[-15:16, -15:16]
    return np.exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / (2.0 * math.pi * sigma * sigma)


def read_image(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read an image as float64 in [0, 255] (2D or (H, W, 3)).

    Palette images convert to RGB; RGBA keeps the first three channels.
    ``size`` optionally resizes with 8-bit semantics (round to uint8 first,
    then bilinear resize).
    """
    with Image.open(path) as im:
        if im.mode == "P":
            im = im.convert("RGB")
        arr = np.asarray(im)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    arr = arr.astype(np.float64)
    if size is not None and arr.shape[:2] != tuple(size):
        u8 = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
        arr = np.asarray(u8.resize((size[1], size[0]), Image.BILINEAR)).astype(np.float64)
    return arr


def _normalize255(img: np.ndarray) -> np.ndarray:
    """Min-max stretch to [0, 255] with MATLAB rounding; all-zero images pass through."""
    lo, hi = float(img.min()), float(img.max())
    if hi == 0.0 and lo == 0.0:
        return img.astype(np.float64)
    return _matlab_round((img - lo) / (hi - lo) * 255.0)


def _hist256(img: np.ndarray) -> np.ndarray:
    """Normalized 256-level grayscale histogram."""
    values = np.clip(np.rint(img), 0, 255).astype(np.int64).ravel()
    return np.bincount(values, minlength=256).astype(np.float64) / values.size


def _blkproc_sum_pow(x: np.ndarray, block: int, power: float) -> np.ndarray:
    """Block-wise power sums (Qcv saliency); right/bottom edges zero-padded."""
    rows, cols = x.shape
    pr, pc = (-rows) % block, (-cols) % block
    if pr or pc:
        x = np.pad(x, ((0, pr), (0, pc)))
    nb_r, nb_c = x.shape[0] // block, x.shape[1] // block
    blocks = x.reshape(nb_r, block, nb_c, block)
    return np.sum(blocks ** power, axis=(1, 3))


def _blkproc_mean_square(x: np.ndarray, block: int) -> np.ndarray:
    """Block-wise mean squares (Qcv distortion)."""
    rows, cols = x.shape
    pr, pc = (-rows) % block, (-cols) % block
    if pr or pc:
        x = np.pad(x, ((0, pr), (0, pc)))
    nb_r, nb_c = x.shape[0] // block, x.shape[1] // block
    blocks = x.reshape(nb_r, block, nb_c, block)
    return np.mean(blocks * blocks, axis=(1, 3))


def _vifb_metric(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray, single) -> float:
    """Adapt a single-channel metric to multi-channel inputs.

    Infrared is always single-channel; RGB fused images are evaluated per
    channel and averaged.
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
    """EN: Shannon entropy of the fused histogram (bits)."""
    p = _hist256(fused)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def _ce_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """CE: mean cross entropy of the sources against the fused image (lower is better)."""
    def ce_pair(a: np.ndarray, f: np.ndarray) -> float:
        p1, p2 = _hist256(_maybe_gray(a)), _hist256(_maybe_gray(f))
        mask = (p1 > 0) & (p2 > 0)
        return float(np.sum(p1[mask] * np.log2(p1[mask] / p2[mask])))

    return (ce_pair(vi, fused) + ce_pair(ir, fused)) / 2.0

def _mi_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """MI: mutual information, summed over both sources.

    Protocol quirk: computed on 256-level joint histograms with *natural*
    logs, not log2. Do not change.
    """
    def mi_pair(a: np.ndarray, b: np.ndarray) -> float:
        def norm(x: np.ndarray) -> np.ndarray:
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo) if hi != lo else np.zeros_like(x)
        ai = np.clip(_matlab_round(norm(_maybe_gray(a)) * 255.0), 0, 255).astype(np.int64).ravel()
        bi = np.clip(_matlab_round(norm(_maybe_gray(b)) * 255.0), 0, 255).astype(np.int64).ravel()
        # joint histogram encoded as ai * 256 + bi
        joint = np.bincount(ai * 256 + bi, minlength=256 * 256).reshape(256, 256).astype(np.float64)
        p_joint = joint / joint.sum()
        ha = np.bincount(ai, minlength=256).astype(np.float64); ha /= ha.sum()
        hb = np.bincount(bi, minlength=256).astype(np.float64); hb /= hb.sum()
        hab = p_joint[p_joint > 0]
        ha_n, hb_n = ha[ha > 0], hb[hb > 0]
        # I(A;B) = H(A) + H(B) - H(A,B), natural logs throughout
        return float(-np.sum(ha_n * np.log(ha_n)) - np.sum(hb_n * np.log(hb_n)) + np.sum(hab * np.log(hab)))
    return mi_pair(vi, fused) + mi_pair(ir, fused)

def _mse_quirk(a: np.ndarray, b: np.ndarray) -> float:
    """Protocol quirk: ``sqrt(SSE) / (m * n)`` — far below standard RMSE.

    Part of calibfuse-metrics-v1; PSNR/RMSE values depend on it. Do not
    change.
    """
    m, n = a.shape
    return float(math.sqrt(np.sum((a - b) ** 2)) / (m * n))


def _psnr_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """PSNR from the averaged quirk MSE of both source directions."""
    mes = (_mse_quirk(_maybe_gray(vi), _maybe_gray(fused)) + _mse_quirk(_maybe_gray(ir), _maybe_gray(fused))) / 2.0
    return float("inf") if mes == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mes)))


def _rmse_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """RMSE from the averaged quirk MSE of both source directions."""
    return (_mse_quirk(_maybe_gray(vi), _maybe_gray(fused)) + _mse_quirk(_maybe_gray(ir), _maybe_gray(fused))) / 2.0


def _ag_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """AG: average gradient."""
    dy, dx = np.gradient(fused)
    s = np.sqrt((dx * dx + dy * dy) / 2.0)
    return float(np.sum(s) / ((fused.shape[0] - 1) * (fused.shape[1] - 1)))


def _ei_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """EI: mean Sobel gradient magnitude."""
    w = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    gx = _imfilter_replicate(fused, w)
    gy = _imfilter_replicate(fused, w.T)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _qabf_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """Qabf (Xydeas-Zivkovic): gradient-based fusion quality.

    Protocol quirks: the equal-gradient branch keeps ``ratio = g_f``
    (common implementations use 1), and gradients below 1e-6 are zeroed.
    """
    h1 = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    h3 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    tg, kg, dg = 0.9994, -15.0, 0.5
    ta, ka, da = 0.9879, -22.0, 0.8

    def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = _conv2(_maybe_gray(img), h3, "same")
        gy = _conv2(_maybe_gray(img), h1, "same")
        gx[np.abs(gx) < 1e-6] = 0.0
        gy[np.abs(gy) < 1e-6] = 0.0
        g = np.sqrt(gx * gx + gy * gy)
        # angle defaults to pi/2 where gx == 0
        ang = np.full(gx.shape, math.pi / 2.0)
        nz = gx != 0
        ang[nz] = np.arctan(gy[nz] / gx[nz])
        return g, ang

    def quality(g_s: np.ndarray, a_s: np.ndarray, g_f: np.ndarray, a_f: np.ndarray) -> np.ndarray:
        # strength ratio measures how well the fused magnitude matches the source
        ratio = np.zeros_like(g_s)
        gt, eq, lt = g_s > g_f, g_s == g_f, g_s < g_f
        ratio[gt] = g_f[gt] / g_s[gt]
        ratio[eq] = g_f[eq]  # historical branch: equal gradients use g_f, not 1
        ratio[lt] = g_s[lt] / g_f[lt]
        align = 1.0 - np.abs(a_s - a_f) / (math.pi / 2.0)
        qg = tg / (1.0 + np.exp(kg * (ratio - dg)))
        qa = ta / (1.0 + np.exp(ka * (align - da)))
        return qg * qa

    ga, aa = sobel(vi)
    gb, ab = sobel(ir)
    gf, af = sobel(fused)
    qa = quality(ga, aa, gf, af)
    qb = quality(gb, ab, gf, af)
    # quality weighted by source gradient strength
    return float(np.sum(qa * ga + qb * gb) / np.sum(ga + gb))


def _scd_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """SCD: sum of correlations of each source with the difference image (Haghighat-Resetti)."""
    def corr(x: np.ndarray, y: np.ndarray) -> float:
        x, y = x - x.mean(), y - y.mean()
        denominator = math.sqrt(np.sum(x * x) * np.sum(y * y))
        return float(np.sum(x * y) / denominator) if denominator else 0.0

    a, b, f = _maybe_gray(vi), _maybe_gray(ir), _maybe_gray(fused)
    return corr(a, f - b) + corr(b, f - a)


def _sd_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """SD: standard deviation of the fused image (contrast)."""
    return float(np.sqrt(np.sum((fused - fused.mean()) ** 2) / fused.size))


def _sf_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """SF: spatial frequency from row/column difference energies."""
    m, n = fused.shape
    rf = np.sum((fused[:, 1:] - fused[:, :-1]) ** 2) / (m * n)
    cf = np.sum((fused[1:, :] - fused[:-1, :]) ** 2) / (m * n)
    return float(math.sqrt(rf + cf))


def _ssim_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """SSIM; protocol quirk: sums both source scores, so it can exceed 1."""
    def ssim_pair(a: np.ndarray, b: np.ndarray) -> float:
        w = _gaussian_window(11, 1.5)
        c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
        a, b = _maybe_gray(a), _maybe_gray(b)
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
    """Qcb (Chen-Blum 2009): CSF filtering -> local contrast -> pooled distortion."""
    im1, im2, imf = _normalize255(_maybe_gray(vi)), _normalize255(_maybe_gray(ir)), _normalize255(_maybe_gray(fused))

    # CSF parameters from the Chen-Blum paper
    f0, f1, a_csf = 15.3870, 1.3456, 0.7622
    k, h_c, p, q, z = 1.0, 1.0, 3.0, 2.0, 0.0001
    rows, cols = im1.shape
    hh, ll = rows / 30.0, cols / 30.0
    u, v = _freqspace_meshgrid(rows, cols)
    r = np.sqrt((ll * u) ** 2 + (hh * v) ** 2)
    # band-pass CSF: difference of two Gaussians
    sd = np.exp(-(r / f0) ** 2) - a_csf * np.exp(-(r / f1) ** 2)

    def csf_filter(im: np.ndarray) -> np.ndarray:
        spec = np.fft.fftshift(np.fft.fft2(im)) * sd
        return np.fft.ifft2(np.fft.ifftshift(spec)).real

    # DoG contrast detection: Gaussians at sigma 2 and 4
    g1, g2 = _gaussian2d_31(2.0), _gaussian2d_31(4.0)

    def contrast(im: np.ndarray) -> np.ndarray:
        return _filter2_same(im, g1) / _filter2_same(im, g2) - 1.0

    def c_pooled(im: np.ndarray) -> np.ndarray:
        c = np.abs(contrast(csf_filter(im)))
        return (k * c ** p) / (h_c * c ** q + z)

    # 0/0 NaN positions never enter the weighted mean below
    with np.errstate(divide="ignore", invalid="ignore"):
        c1p, c2p, cfp = c_pooled(im1), c_pooled(im2), c_pooled(imf)
        # preservation per direction: smaller/larger ratio keeps values in (0, 1]
        mask = c1p < cfp
        q1f = (c1p / cfp) * mask + (cfp / c1p) * (~mask)
        mask = c2p < cfp
        q2f = (c2p / cfp) * mask + (cfp / c2p) * (~mask)

    # saliency weights: normalized shares of pooled contrast
    ramda1 = (c1p * c1p) / (c1p * c1p + c2p * c2p)
    ramda2 = (c2p * c2p) / (c1p * c1p + c2p * c2p)
    return float(np.mean(ramda1 * q1f + ramda2 * q2f))


def _qcv_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """Qcv (Chen-Varma): saliency-weighted CSF-filtered block distortion (lower is better)."""
    im1, im2, imf = _normalize255(_maybe_gray(vi)), _normalize255(_maybe_gray(ir)), _normalize255(_maybe_gray(fused))

    flt1 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    flt2 = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])

    def grad_mag(im: np.ndarray) -> np.ndarray:
        gx, gy = _filter2_same(im, flt1), _filter2_same(im, flt2)
        return np.sqrt(gx * gx + gy * gy)

    # block window 16, saliency power 5
    window, alpha = 16, 5
    ramda1 = _blkproc_sum_pow(grad_mag(im1), window, alpha)
    ramda2 = _blkproc_sum_pow(grad_mag(im2), window, alpha)

    rows, cols = im1.shape
    u, v = _freqspace_meshgrid(rows, cols)
    r = np.sqrt((cols / 8.0 * u) ** 2 + (rows / 8.0 * v) ** 2)

    # human modulation transfer function (Chen-Varma)
    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-((0.144 * r) ** 1.1))

    def csf_filter(diff: np.ndarray) -> np.ndarray:
        spec = np.fft.fftshift(np.fft.fft2(diff)) * theta_m
        return np.fft.ifft2(np.fft.ifftshift(spec)).real

    # distortion = block mean square of the CSF-filtered source difference
    d1 = _blkproc_mean_square(csf_filter(im1 - imf), window)
    d2 = _blkproc_mean_square(csf_filter(im2 - imf), window)
    return float(np.sum(ramda1 * d1 + ramda2 * d2) / np.sum(ramda1 + ramda2))


def _vif_single(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
    """VIF (Sheikh-Bovik); protocol quirk: sums both source scores.

    Standard 4-scale implementation with Gaussian-window local statistics
    and 2x downsampling between scales.
    """
    def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        radius = (size - 1) / 2.0
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        # truncate tiny weights, matching MATLAB fspecial
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
            N = 2 ** (5 - scale) + 1
            win = np.rot90(gaussian_kernel(N, N / 5.0), 2)
            if scale > 1:
                # low-pass, then decimate by 2
                ref = _conv2(ref, win, "valid")[::2, ::2]
                dist = _conv2(dist, win, "valid")[::2, ::2]
            mu1 = _conv2(ref, win, "valid")
            mu2 = _conv2(dist, win, "valid")
            sigma1_sq = _conv2(ref * ref, win, "valid") - mu1 * mu1
            sigma2_sq = _conv2(dist * dist, win, "valid") - mu2 * mu2
            sigma12 = _conv2(ref * dist, win, "valid") - mu1 * mu2
            # guard against negative variances from numerical error
            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0
            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            # degenerate-region handling (standard VIF branches)
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
    """Wrap a single-channel metric into a multi-channel-capable entry point."""
    def metric(vi: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> float:
        return _vifb_metric(vi, ir, fused, single)
    return metric


# Public metric entry points (RGB fused images are handled per channel).
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
    """Compute the requested metrics; returns ``{name: value}``."""
    return {name: METRIC_FUNCS[name](vi, ir, fused) for name in metrics}
