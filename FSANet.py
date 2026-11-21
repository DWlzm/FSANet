"""FSANet: Frequency-Spatial Attention Network with Dual Color Space Fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


def rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 4 and x.size(1) == 3
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    maxc = torch.max(x, dim=1, keepdim=True)[0]
    minc = torch.min(x, dim=1, keepdim=True)[0]
    diff = maxc - minc
    eps = 1e-8
    safe = diff + eps

    h = torch.zeros_like(maxc)
    mask_r = (maxc == r) & (diff > eps)
    mask_g = (maxc == g) & (diff > eps) & ~mask_r
    mask_b = (maxc == b) & (diff > eps) & ~mask_r & ~mask_g
    h = torch.where(mask_r, ((g - b) / safe) % 6, h)
    h = torch.where(mask_g, (b - r) / safe + 2, h)
    h = torch.where(mask_b, (r - g) / safe + 4, h)
    h = h / 6.0

    s = torch.where(maxc > eps, diff / (maxc + eps), torch.zeros_like(diff))
    v = maxc
    return torch.cat([h, s, v], dim=1)


def cbr(in_c, out_c, k=3):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return x * self.sigmoid(self.mlp(avg) + self.mlp(mx))


class DBFM(nn.Module):
    def __init__(self, in_channels=3, n_filters=32):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, n_filters, 1, bias=False)

        self.rgb_conv = nn.Sequential(
            cbr(in_channels, n_filters, 3),
            cbr(n_filters, n_filters, 3),
        )
        self.rgb_sa = SpatialAttention(kernel_size=7)

        self.hsv_conv = nn.Sequential(
            cbr(in_channels, n_filters, 5),
            cbr(n_filters, n_filters, 5),
        )
        self.hsv_ca = ChannelAttention(n_filters, reduction=16)

        self.crc = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, 2, 3, padding=1, bias=False),
        )

    def forward(self, x):
        x_feat = self.stem(x)
        f_rgb = self.rgb_sa(self.rgb_conv(x))
        f_hsv = self.hsv_ca(self.hsv_conv(rgb_to_hsv(x)))

        fg = torch.cat([
            F.adaptive_avg_pool2d(f_rgb, 1),
            F.adaptive_avg_pool2d(f_hsv, 1),
        ], dim=1)
        v = torch.softmax(self.crc(fg), dim=1)
        v_rgb, v_hsv = v[:, 0:1], v[:, 1:2]
        return v_rgb * f_rgb + v_hsv * f_hsv + x_feat


class DFSAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = nn.Sequential(
            cbr(channels, channels, 3),
            cbr(channels, channels, 3),
        )
        # 空间分支：depthwise + pointwise
        self.dw_conv = nn.Conv2d(channels, channels, 7, padding=3,
                                 groups=channels, bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)
        # 频域分支：直接在频谱尺寸上做，real/imag 分别调制
        self.freq_conv = nn.Conv2d(channels * 2, channels * 2, 1, bias=False)
        # 门控：逐像素，看两个分支内容
        self.gate_conv = nn.Conv2d(channels * 2, 1, 1, bias=True)

    def forward(self, F_in):
        B, C, H, W = F_in.shape

        F_init = self.local(F_in)

        # 空间分支
        F_s = F_init * torch.sigmoid(self.pw_conv(self.dw_conv(F_init)))

        # 频域分支：无插值，直接频谱尺寸调制
        fft_x = torch.fft.rfft2(F_init, norm='ortho')       # (B,C,H,W//2+1)
        F_freq = torch.cat([fft_x.real, fft_x.imag], dim=1) # (B,2C,H,W//2+1)
        a = self.freq_conv(F_freq)                          # (B,2C,H,W//2+1)
        a_real, a_imag = a[:, :C], a[:, C:]
        # 以 1 为中心的增益，允许增强/衰减
        gain_r = 1.0 + torch.tanh(a_real)
        gain_i = 1.0 + torch.tanh(a_imag)
        modulated = torch.complex(fft_x.real * gain_r,
                                  fft_x.imag * gain_i)
        F_f = torch.fft.irfft2(modulated, s=(H, W), norm='ortho')
        F_f = F_f + F_init   # 残差，降低频域分支独立承担输出的压力

        # 逐像素门控
        g = torch.sigmoid(self.gate_conv(torch.cat([F_s, F_f], dim=1)))  # (B,1,H,W)
        return g * F_s + (1 - g) * F_f + F_in   # 外层再加输入残差

class SharedChannelAttention(nn.Module):
    def __init__(self, c_high, c_low, reduction=16):
        super().__init__()
        c_mid = max((c_high + c_low) // reduction, 8)
        self.fc_shared = nn.Linear(c_high + c_low, c_mid)
        self.bn_shared = nn.BatchNorm1d(c_mid)
        self.fc_high = nn.Linear(c_mid, c_high)
        self.fc_low = nn.Linear(c_mid, c_low)
        self.bn_high = nn.BatchNorm1d(c_high)
        self.bn_low = nn.BatchNorm1d(c_low)

    def forward(self, F_high, F_low):
        z_high = F.adaptive_avg_pool2d(F_high, 1).flatten(1)
        z_low = F.adaptive_avg_pool2d(F_low, 1).flatten(1)
        zs = self.bn_shared(self.fc_shared(torch.cat([z_high, z_low], dim=1)))
        a_high = torch.sigmoid(self.bn_high(self.fc_high(zs)))
        a_low = torch.sigmoid(self.bn_low(self.fc_low(zs)))
        return (F_high * a_high.unsqueeze(-1).unsqueeze(-1),
                F_low * a_low.unsqueeze(-1).unsqueeze(-1))


class SMSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv7 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.fuse = nn.Conv2d(2, 1, 1, bias=False)

    def _spatial_attention(self, F_in):
        cap = F_in.mean(dim=1, keepdim=True)
        cmp = F_in.max(dim=1, keepdim=True)[0]
        pool = torch.cat([cap, cmp], dim=1)
        f3 = self.conv3(pool)
        f7 = self.conv7(pool)
        attn = torch.sigmoid(self.fuse(torch.cat([f3, f7], dim=1)))
        return F_in * attn

    def forward(self, F_high, F_low):
        return self._spatial_attention(F_high), self._spatial_attention(F_low)


class DBAGM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cbr_sca = cbr(channels, channels, 1)
        self.cbr_smsa = cbr(channels, channels, 1)
        self.cbr_joint = cbr(channels * 2, channels, 1)
        self.gate = nn.Conv2d(channels, 2, 1, bias=True)
        self.out_conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, F_sca, F_smsa):
        a = self.cbr_sca(F_sca)
        b = self.cbr_smsa(F_smsa)
        joint = self.cbr_joint(torch.cat([a, b], dim=1))
        gates = torch.softmax(self.gate(joint), dim=1)
        return self.out_conv(gates[:, 0:1] * F_sca + gates[:, 1:2] * F_smsa)


class ABM(nn.Module):
    def __init__(self, c_high, c_low):
        super().__init__()
        self.sca = SharedChannelAttention(c_high, c_low)
        self.smsa = SMSA()
        self.dbagm_high = DBAGM(c_high)
        self.dbagm_low = DBAGM(c_low)

    def forward(self, F_high, F_low):
        return F_high,F_low
        h_sca, l_sca = self.sca(F_high, F_low)
        h_smsa, l_smsa = self.smsa(F_high, F_low)
        return self.dbagm_high(h_sca, h_smsa), self.dbagm_low(l_sca, l_smsa)


class FSANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,
                 n_filters=32, use_abm=True,
                 enc4_layers=2, bottleneck_layers=7):
        super().__init__()
        f = n_filters
        self.n_filters = n_filters
        self.use_abm = use_abm
        self.enc4_layers = enc4_layers
        self.bottleneck_layers = bottleneck_layers

        self.dbfm = DBFM(in_channels, f)

        self.enc2 = nn.Sequential(
            cbr(f, f * 2, 3),
            cbr(f * 2, f * 2, 3),
        )
        self.enc3 = nn.Sequential(
            cbr(f * 2, f * 4, 3),
            cbr(f * 4, f * 4, 3),
        )

        enc4_list = [cbr(f * 4, f * 8, 3)]
        for _ in range(enc4_layers - 1):
            enc4_list.append(cbr(f * 8, f * 8, 3))
        self.enc4 = nn.Sequential(*enc4_list)

        self.pool = nn.MaxPool2d(2)

        self.dfsam1 = DFSAM(f)
        self.dfsam2 = DFSAM(f * 2)
        self.dfsam3 = DFSAM(f * 4)
        self.dfsam4 = DFSAM(f * 8)

        f16 = f * 16
        bn_layers = [cbr(f * 8, f16, 3)]
        for _ in range(bottleneck_layers - 1):
            bn_layers.append(cbr(f16, f16, 3))
        self.bottleneck = nn.Sequential(*bn_layers)

        if use_abm:
            self.abm12 = ABM(f, f * 2)
            self.abm34 = ABM(f * 4, f * 8)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = nn.Sequential(
            cbr(f * 16, f * 8, 3),
            cbr(f * 8, f * 8, 3),
        )
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            cbr(f * 8, f * 4, 3),
            cbr(f * 4, f * 4, 3),
        )
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            cbr(f * 4, f * 2, 3),
            cbr(f * 2, f * 2, 3),
        )
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = nn.Sequential(
            cbr(f * 2, f, 3),
            cbr(f, f, 3),
        )

        self.out_conv = nn.Conv2d(f, out_channels, 1)
        self.final_act = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.dfsam1(self.dbfm(x))
        e2 = self.dfsam2(self.enc2(self.pool(e1)))
        e3 = self.dfsam3(self.enc3(self.pool(e2)))
        e4 = self.dfsam4(self.enc4(self.pool(e3)))
        b = self.bottleneck(self.pool(e4))

        if self.use_abm:
            e1, e2 = self.abm12(e1, e2)
            e3, e4 = self.abm34(e3, e4)

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.final_act(self.out_conv(d1))
        return out, None


if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = FSANet(in_channels=3, out_channels=3,
                   n_filters=32, bottleneck_layers=7, enc4_layers=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FSANet] Config: n_filters=32, bottleneck_layers=7, enc4_layers=2, ABM x2")
    print(f"[FSANet] Params: {n_params / 1e6:.2f} M")

    model.train()
    x = torch.rand(2, 3, 256, 256, device=device)
    y, _ = model(x)
    print(f"[FSANet] train  x={tuple(x.shape)} -> y={tuple(y.shape)}")
    assert y.shape == x.shape
    assert y.min() >= 0 and y.max() <= 1
    assert torch.isfinite(y).all()

    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 3, 256, 256, device=device)
        y, _ = model(x)
        print(f"[FSANet] eval   x={tuple(x.shape)} -> y={tuple(y.shape)}")
        assert y.shape == x.shape

        x2 = torch.rand(1, 3, 240, 320, device=device)
        y2, _ = model(x2)
        print(f"[FSANet] odd    x={tuple(x2.shape)} -> y={tuple(y2.shape)}")
        assert y2.shape == x2.shape

    model.train()
    x = torch.rand(2, 3, 128, 128, device=device)
    y, _ = model(x)
    loss = y.mean()
    loss.backward()
    print(f"[FSANet] backward OK, loss={loss.item():.4f}")

    print("\n----- Module tests -----")

    dbfm = DBFM(3, 32).to(device).train()
    x = torch.rand(2, 3, 64, 64, device=device)
    y = dbfm(x)
    print(f"[DBFM]  {tuple(x.shape)} -> {tuple(y.shape)}")
    assert y.shape == (2, 32, 64, 64)

    for c in [32, 64, 128, 256]:
        dfsam = DFSAM(c).to(device).train()
        x = torch.rand(2, c, 32, 32, device=device)
        y = dfsam(x)
        print(f"[DFSAM] C={c:3d}  {tuple(x.shape)} -> {tuple(y.shape)}")
        assert y.shape == x.shape

    sca = SharedChannelAttention(32, 64).to(device).train()
    xh = torch.rand(2, 32, 64, 64, device=device)
    xl = torch.rand(2, 64, 32, 32, device=device)
    yh, yl = sca(xh, xl)
    print(f"[SCA]   high {tuple(xh.shape)}->{tuple(yh.shape)}, "
          f"low {tuple(xl.shape)}->{tuple(yl.shape)}")
    assert yh.shape == xh.shape and yl.shape == xl.shape

    smsa = SMSA().to(device).train()
    yh, yl = smsa(xh, xl)
    print(f"[SMSA]  high {tuple(yh.shape)}, low {tuple(yl.shape)}")
    assert yh.shape == xh.shape and yl.shape == xl.shape

    db = DBAGM(32).to(device).train()
    a = torch.rand(2, 32, 64, 64, device=device)
    b_ = torch.rand(2, 32, 64, 64, device=device)
    y = db(a, b_)
    print(f"[DBAGM] 2x{tuple(a.shape)} -> {tuple(y.shape)}")
    assert y.shape == a.shape

    abm = ABM(32, 64).to(device).train()
    xh = torch.rand(2, 32, 64, 64, device=device)
    xl = torch.rand(2, 64, 32, 32, device=device)
    yh, yl = abm(xh, xl)
    print(f"[ABM]   high {tuple(xh.shape)}->{tuple(yh.shape)}, "
          f"low {tuple(xl.shape)}->{tuple(yl.shape)}")
    assert yh.shape == xh.shape and yl.shape == xl.shape

    print("\nAll tensor tests passed")