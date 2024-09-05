import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ndim):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_ch, out_ch),
            nn.MaxPool2d((2, 1)),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_ndim, out_ch),
        )

    def forward(self, x, t):
        x = self.conv(x)
        b, ch, seq_len, pt = x.size()
        t = self.emb_layer(t).view(b, ch, 1, 1).repeat(1, 1, seq_len, pt)
        return x + t


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, size, hidden_ndim):
        super().__init__()

        self.up = nn.Upsample(size, mode="bilinear", align_corners=True)
        self.conv = Conv(in_ch, out_ch)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_ndim, out_ch),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        b, ch, seq_len, pt = x.size()
        t = self.emb_layer(t).view(b, ch, 1, 1).repeat(1, 1, seq_len, pt)
        return x + t


class UNet(nn.Module):
    def __init__(self, config, num_classes, skel_size):
        super().__init__()
        self.seq_len = config.seq_len - 1
        self.hidden_ndim = config.hidden_ndim
        size = (self.seq_len, skel_size[0] * skel_size[1])

        self.emb_y = nn.Embedding(num_classes, self.hidden_ndim)

        self.conv_in = Conv(1, 64)
        self.down1 = Down(64, 128, config.hidden_ndim)

        self.bot1 = Conv(128, 256)
        self.bot2 = Conv(256, 256)
        self.bot3 = Conv(256, 128)

        self.up1 = Up(128 + 64, 128, size, config.hidden_ndim)
        self.conv_out = nn.Sequential(Conv(128, 64), Conv(64, 1))

    def pos_encoding(self, t, hidden_ndim):
        freq = 10000 ** (torch.arange(0, hidden_ndim, 2).to(t.device) / hidden_ndim)
        pos_enc_a = torch.sin(t.repeat(1, hidden_ndim // 2) / freq)
        pos_enc_b = torch.cos(t.repeat(1, hidden_ndim // 2) / freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, t, x, y):
        b, seq_len, pt, d = x.size()

        t = self.pos_encoding(t.view(b, 1), self.hidden_ndim)
        y = self.emb_y(y)
        t = t + y

        x = x.view(b, 1, seq_len, pt * d)
        x1 = self.conv_in(x)
        x2 = self.down1(x1, t)

        x2 = self.bot1(x2)
        x2 = self.bot2(x2)
        x2 = self.bot3(x2)

        x = self.up1(x2, x1, t)
        x = self.conv_out(x)

        return x.view(b, seq_len, pt, d)
