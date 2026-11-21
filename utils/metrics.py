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


def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """Asymmetric alpha-trimmed mean."""
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    weight = (1/(K-T_a_L-T_a_R))
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag) 
    return mag

def eme(x, window_size):
    """Enhancement measure estimation."""
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def _uism(x):
    """Underwater Image Sharpness Measure."""
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
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
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
    return mu-x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
    alpha = 1
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
    return w*val

def getUIQM(x):
    """UIQM score, coefficients from the original paper."""
    x = x.astype(np.float32)
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm

def getUCIQE(rgb_in):
    rgb_in = cv2.normalize(rgb_in, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if len(rgb_in.shape) == 4:
        rgb_in = rgb_in[0]
    if rgb_in.shape[0] == 3:  # (channel, height, width)
        rgb_in = np.transpose(rgb_in, (1, 2, 0))

    bgr_in = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2BGR)
    (l,a,b) = cv2.split(bgr_in)
    Chroma = np.sqrt(a*a + b*b)
    StdVarianceChroma = np.std(np.reshape(Chroma[:,:],(-1,1)))

    hsv = skimage.color.rgb2hsv(rgb_in)
    Saturation = hsv[:,:,2]
    MeanSaturation = np.mean(np.reshape(Saturation[:,:],(-1,1)))

    ContrastLuminance = max(np.reshape(l[:,:],(-1,1))) - min(np.reshape(l[:,:],(-1,1)))
    UCIQE = 0.4680 * StdVarianceChroma + 0.2745 * ContrastLuminance + 0.2576 * MeanSaturation
    return float(UCIQE)

def calculate_pcqi(img1, img2):
    """Patch-based contrast quality index."""
    if len(img1.shape) == 4:
        img1 = img1[0]
        img2 = img2[0]

    if img1.shape[0] == 3:  # (channel, height, width)
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray1, gray2 = img1, img2

    patch_size = 8
    h, w = gray1.shape
    pcqi_values = []

    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch1 = gray1[i:i+patch_size, j:j+patch_size]
            patch2 = gray2[i:i+patch_size, j:j+patch_size]

            mean1, std1 = np.mean(patch1), np.std(patch1)
            mean2, std2 = np.mean(patch2), np.std(patch2)

            if std1 + std2 > 0:
                contrast_sim = 2 * std1 * std2 / (std1**2 + std2**2)
            else:
                contrast_sim = 1.0

            if mean1 + mean2 > 0:
                luminance_sim = 2 * mean1 * mean2 / (mean1**2 + mean2**2)
            else:
                luminance_sim = 1.0

            pcqi_patch = contrast_sim * luminance_sim
            pcqi_values.append(pcqi_patch)

    pcqi = np.mean(pcqi_values)

    return pcqi

def calculate_all_metrics(img1, img2):
    """All metrics for one image pair, values in [0, 1]."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().detach().numpy()

    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)

    metrics = {}

    metrics['psnr'] = psnr(img1, img2, data_range=1.0)

    # SSIM window must fit the image and be odd
    if len(img1.shape) == 4:
        min_dim = min(img1.shape[2:4])
    elif len(img1.shape) == 3:
        min_dim = min(img1.shape[1:3])
    elif len(img1.shape) == 2:
        min_dim = min(img1.shape)
    else:
        min_dim = 7

    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3

    if len(img1.shape) == 4:
        img1_3d = img1[0]
        img2_3d = img2[0]
    else:
        img1_3d = img1
        img2_3d = img2

    metrics['ssim'] = ssim(img1_3d, img2_3d, data_range=1.0, channel_axis=0, win_size=win_size)
    metrics['mse'] = mse(img1, img2)
    metrics['uiqm'] = getUIQM(img2_uint8[0] if len(img2_uint8.shape) == 4 else img2_uint8)
    metrics['uciqe'] = getUCIQE(img2_uint8[0] if len(img2_uint8.shape) == 4 else img2_uint8)
    metrics['pcqi'] = calculate_pcqi(img1_uint8[0] if len(img1_uint8.shape) == 4 else img1_uint8, 
                                   img2_uint8[0] if len(img2_uint8.shape) == 4 else img2_uint8)

    return metrics


def calculate_composite_score(metrics):
    """Weighted composite score used to pick the best checkpoint."""
    weights = get_metric_weights()
    baselines = get_metric_baselines()

    psnr_score = min(metrics['psnr'] / baselines['psnr'], 1.0)
    ssim_score = metrics['ssim']
    uiqm_score = min(metrics['uiqm'] / baselines['uiqm'], 1.0)
    uciqe_score = min(metrics['uciqe'] / baselines['uciqe'], 1.0)
    pcqi_score = min(metrics['pcqi'] / baselines['pcqi'], 1.0)
    mse_score = max(0, 1.0 - metrics['mse'] / baselines['mse'])

    composite_score = (
        weights['psnr'] * psnr_score +
        weights['ssim'] * ssim_score +
        weights['uiqm'] * uiqm_score +
        weights['uciqe'] * uciqe_score +
        weights['pcqi'] * pcqi_score +
        weights['mse'] * mse_score
    )

    return composite_score
