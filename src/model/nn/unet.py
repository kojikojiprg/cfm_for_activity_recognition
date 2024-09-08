import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_nch, out_nch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_nch, out_nch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_nch),
            nn.SiLU(),
            nn.Conv2d(out_nch, out_nch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_nch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_nch, out_nch, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_nch, out_nch),
            nn.AvgPool2d(kernel_size, stride=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_nch, out_nch, kernel_size):
        super().__init__()
        self.convtrans = nn.ConvTranspose2d(in_nch // 2, in_nch // 2, kernel_size)
        self.conv = Conv(in_nch, out_nch)

    def forward(self, x, skip_x):
        x = self.convtrans(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, config, num_classes, skel_size):
        super().__init__()
        self.lmd_pe = config.lmd_pe
        self.lmd_y = config.lmd_y
        nch = config.hidden_nch

        self.emb_y = nn.Embedding(num_classes, skel_size[0] * skel_size[1])

        self.conv_in = Conv(1, nch // 8)
        self.down1 = Down(nch // 8, nch // 4, (9, 2))
        self.conv1 = Conv(nch // 4, nch // 4)
        self.down2 = Down(nch // 4, nch // 2, (7, 1))
        self.conv2 = Conv(nch // 2, nch // 2)
        self.down3 = Down(nch // 2, nch // 2, (5, 1))

        self.bot1 = Conv(nch // 2, nch)
        self.bot2 = Conv(nch, nch)
        self.bot3 = Conv(nch, nch // 2)

        self.up4 = Up(nch, nch // 4, (5, 1))
        self.conv4 = Conv(nch // 4, nch // 4)
        self.up5 = Up(nch // 2, nch // 8, (7, 1))
        self.conv5 = Conv(nch // 8, nch // 8)
        self.up6 = Up(nch // 4, nch // 8, (9, 2))
        self.conv_out = Conv(nch // 8, 1)

    def pos_encoding(self, t, size):
        b, pt, d = size
        ndim = pt * d
        freq = 10000 ** (torch.arange(0, ndim, 2).to(t.device) / ndim)
        if ndim % 2 == 0:
            pos_enc_a = torch.sin(t.repeat(1, ndim // 2) / freq)
            pos_enc_b = torch.cos(t.repeat(1, ndim // 2) / freq)
        else:
            pos_enc_a = torch.sin(t.repeat(1, ndim // 2 + 1) / freq)
            pos_enc_b = torch.cos(t.repeat(1, ndim // 2) / freq[:-1])
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, t, x, y):
        b, pt, d = x.size()
        t = self.pos_encoding(t.view(b, 1), x.size())
        t = t.view(b, 1, pt, d) * self.lmd_pe

        y = self.emb_y(y)
        y = y.view(b, 1, pt, d) * self.lmd_y

        x = x.view(b, 1, pt, d)
        x = x + y + t

        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up4(x4, x3)
        x = self.conv4(x)
        x = self.up5(x, x2)
        x = self.conv5(x)
        x = self.up6(x, x1)
        x = self.conv_out(x)

        return x.view(b, pt, d)
