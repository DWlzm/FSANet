"""Evaluation metrics: PSNR / SSIM / MSE / UIQM / UCIQE / PCQI and the composite score."""
import numpy as np
import cv2
import math
import torch
import skimage
from skimage import color
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from utils.model_config import get_metric_weights, get_metric_baselines


# ---------------------------------------------------------------------------
# Layout / dtype helpers
# ---------------------------------------------------------------------------
def _to_hwc_uint8(img):
    """Convert tensor/ndarray (any layout, float[0,1] or uint8) to HWC uint8 RGB."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = np.asarray(img)

    if img.ndim == 4:
        img = img[0]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        # CHW -> HWC if first axis looks like channels
        if img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[0] == 1:
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0 + 0.5).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# UIQM components
# ---------------------------------------------------------------------------
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """Asymmetric alpha-trimmed mean (vectorized)."""
    x = np.sort(np.asarray(x).ravel())
    K = x.size
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    denom = K - T_a_L - T_a_R
    if denom <= 0:
        return float(np.mean(x)) if K > 0 else 0.0
    s = int(T_a_L)
    e = int(K - T_a_R)
    return float(x[s:e].sum() / denom)


def s_a(x, mu):
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean((x - mu) ** 2))


def _uicm(x):
    R = x[:, :, 0].astype(np.float64)
    G = x[:, :, 1].astype(np.float64)
    B = x[:, :, 2].astype(np.float64)
    RG = R - G
    YB = ((R + G) / 2.0) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt(mu_a_RG ** 2 + mu_a_YB ** 2)
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)


def sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    m = float(np.max(mag))
    if m > 0:
        mag = mag * (255.0 / m)
    return mag


def eme(x, window_size):
    """Enhancement measure estimation."""
    x = np.asarray(x, dtype=np.float64)
    k1 = x.shape[1] / window_size
    k2 = x.shape[0] / window_size
    w = 2.0 / (k1 * k2)
    x = x[: int(window_size * k2), : int(window_size * k1)]
    val = 0.0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k * window_size: window_size * (k + 1),
                      l * window_size: window_size * (l + 1)]
            max_ = float(np.max(block))
            min_ = float(np.min(block))
            if min_ == 0.0 or max_ == 0.0:
                continue
            val += math.log(max_ / min_)
    return w * val


def _uism(x):
    """Underwater Image Sharpness Measure."""
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.114          # was 0.144 (typo)
    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)


def _uiconm(x, window_size):
    x = np.asarray(x, dtype=np.float64)
    k1 = x.shape[1] / window_size
    k2 = x.shape[0] / window_size
    w = -1.0 / (k1 * k2)
    x = x[: int(window_size * k2), : int(window_size * k1)]
    alpha = 1
    val = 0.0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k * window_size: window_size * (k + 1),
                      l * window_size: window_size * (l + 1), :]
            max_ = float(np.max(block))
            min_ = float(np.min(block))
            top = max_ - min_
            bot = max_ + min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                continue
            val += alpha * math.pow(top / bot, alpha) * math.log(top / bot)
    return w * val


def getUIQM(x):
    """UIQM score, coefficients from the original paper."""
    x = _to_hwc_uint8(x).astype(np.float32)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 10)
    return (c1 * uicm) + (c2 * uism) + (c3 * uiconm)


# ---------------------------------------------------------------------------
# UCIQE
# ---------------------------------------------------------------------------
def getUCIQE(rgb_in):
    """
    UCIQE (Underwater Color Image Quality Evaluation), Yang & Sowmya 2015.
    UCIQE = 0.4680 * std(chroma) + 0.2745 * (max(L)-min(L)) + 0.2576 * mean(S)
    All three terms normalized to [0, 1].
    """
    rgb_in = _to_hwc_uint8(rgb_in)

    # LAB: L,a,b all normalized to a comparable [0,1]-ish scale
    lab = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2LAB)   # uint8: L∈[0,255], a,b∈[0,255]
    L = lab[:, :, 0].astype(np.float32) / 255.0
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0

    chroma = np.sqrt(a * a + b * b) / 181.0          # ~[0,1]
    std_chroma = float(np.std(chroma))

    con_l = float(L.max() - L.min())                 # [0,1]

    hsv = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2HSV)    # uint8: H∈[0,179], S,V∈[0,255]
    S = hsv[:, :, 1].astype(np.float32) / 255.0      # [0,1]
    mean_sat = float(np.mean(S))

    uciqe = 0.4680 * std_chroma + 0.2745 * con_l + 0.2576 * mean_sat
    return float(uciqe)


# ---------------------------------------------------------------------------
# PCQI
# ---------------------------------------------------------------------------
def calculate_pcqi(img1, img2):
    """Patch-based contrast quality index (approximate, patch_size=8)."""
    img1 = _to_hwc_uint8(img1)
    img2 = _to_hwc_uint8(img2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64)

    patch_size = 8
    h, w = gray1.shape
    if h < patch_size or w < patch_size:
        return float("nan")

    pcqi_values = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            p1 = gray1[i:i + patch_size, j:j + patch_size]
            p2 = gray2[i:i + patch_size, j:j + patch_size]
            mean1, std1 = float(np.mean(p1)), float(np.std(p1))
            mean2, std2 = float(np.mean(p2)), float(np.std(p2))

            denom_c = std1 ** 2 + std2 ** 2
            contrast_sim = (2 * std1 * std2 / denom_c) if denom_c > 0 else 1.0

            denom_l = mean1 ** 2 + mean2 ** 2
            luminance_sim = (2 * mean1 * mean2 / denom_l) if denom_l > 0 else 1.0

            pcqi_values.append(contrast_sim * luminance_sim)

    return float(np.mean(pcqi_values)) if pcqi_values else float("nan")


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def calculate_all_metrics(img1, img2):
    """All metrics for one image pair. img1/img2: float [0,1], any layout."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().detach().numpy()

    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    metrics = {}

    metrics['psnr'] = float(psnr(img1, img2, data_range=1.0))
    metrics['mse'] = float(mse(img1, img2))

    # ---- SSIM: reduce to 3D, compute min spatial dim ----
    a = img1[0] if img1.ndim == 4 else img1
    b = img2[0] if img2.ndim == 4 else img2

    # channel_axis: 0 for CHW, -1 for HWC
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        channel_axis = 0
        min_dim = min(a.shape[1], a.shape[2])
    elif a.ndim == 3 and a.shape[-1] in (1, 3):
        channel_axis = -1
        min_dim = min(a.shape[0], a.shape[1])
    else:  # 2D
        channel_axis = None
        min_dim = min(a.shape[0], a.shape[1])

    if min_dim < 3:
        metrics['ssim'] = float("nan")
    else:
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        win_size = max(3, win_size)
        metrics['ssim'] = float(ssim(a, b, data_range=1.0,
                                     channel_axis=channel_axis,
                                     win_size=win_size))

    # ---- UIQM / UCIQE / PCQI: functions internally handle layout+dtype ----
    metrics['uiqm'] = float(getUIQM(img2))
    metrics['uciqe'] = float(getUCIQE(img2))
    metrics['pcqi'] = float(calculate_pcqi(img1, img2))

    return metrics


def calculate_composite_score(metrics):
    """Weighted composite score used to pick the best checkpoint."""
    weights = get_metric_weights()
    baselines = get_metric_baselines()

    def _safe_div(v, base, cap_high=True):
        if base is None or base == 0 or not np.isfinite(base):
            return 0.0
        r = v / base
        if not np.isfinite(r):
            return 0.0
        return min(r, 1.0) if cap_high else max(0.0, r)

    psnr_score = _safe_div(metrics['psnr'], baselines['psnr'], True)
    ssim_score = metrics['ssim'] if np.isfinite(metrics['ssim']) else 0.0
    uiqm_score = _safe_div(metrics['uiqm'], baselines['uiqm'], True)
    uciqe_score = _safe_div(metrics['uciqe'], baselines['uciqe'], True)
    pcqi_score = _safe_div(metrics['pcqi'], baselines['pcqi'], True)
    mse_score = max(0.0, 1.0 - _safe_div(metrics['mse'], baselines['mse'], False))

    composite_score = (
        weights['psnr'] * psnr_score +
        weights['ssim'] * ssim_score +
        weights['uiqm'] * uiqm_score +
        weights['uciqe'] * uciqe_score +
        weights['pcqi'] * pcqi_score +
        weights['mse'] * mse_score
    )
    return float(composite_score)
