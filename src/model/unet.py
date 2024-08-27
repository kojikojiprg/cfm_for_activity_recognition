import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.ln = nn.LayerNorm([channels])
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(1, 2)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.SiLU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.silu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            Conv(in_channels, in_channels, residual=True),
            Conv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, size, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(size, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            Conv(in_channels, in_channels, residual=True),
            Conv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, config, size, num_classes=None):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim
        size = size[0] * size[1]

        self.inc = Conv(1, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)

        self.bot1 = Conv(256, 512)
        self.bot2 = Conv(512, 512)
        self.bot3 = Conv(512, 256)

        self.up1 = Up(256 + 128, 128, size // 2)
        self.sa3 = SelfAttention(128)
        self.up2 = Up(128 + 64, 64, size)
        self.sa4 = SelfAttention(64)
        self.outc = nn.Conv1d(64, 1, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.hidden_ndim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, tau, v, y=None):
        b, pt, d = v.size()
        v = v.view(b, 1, pt * d)

        tau = tau.unsqueeze(-1).type(torch.float)
        tau = self.pos_encoding(tau, self.hidden_ndim)

        if y is not None:
            tau += self.label_emb(y)

        x1 = self.inc(v)

        x2 = self.down1(x1, tau)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, tau)
        x3 = self.sa2(x3)

        x3 = self.bot1(x3)
        x3 = self.bot2(x3)
        x3 = self.bot3(x3)

        v = self.up1(x3, x2, tau)
        v = self.sa3(v)
        v = self.up2(v, x1, tau)
        v = self.sa4(v)

        output = self.outc(v)
        output = output.view(b, pt, d)
        return output
