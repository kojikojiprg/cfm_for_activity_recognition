import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchdyn.core import NeuralODE

from .cfm import ConditionalFlowMatcher
from .nn import UNet


class FlowMatching(LightningModule):
    def __init__(self, config, n_clusters=120, skel_size=(25, 3), mag=1):
        super().__init__()
        self.config = config
        # self.seq_len = config.seq_len
        # self.steps = config.steps
        # self.sigma = config.sigma
        self.n_clusters = n_clusters
        self.skel_size = skel_size
        self.mag = mag
        self.cfm = None
        self.net = None

    def configure_model(self):
        if self.cfm is None:
            self.cfm = ConditionalFlowMatcher(self.config)
        if self.net is None:
            self.net = UNet(self.config, self.n_clusters, self.skel_size)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), self.config.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, self.config.t_max, self.config.lr_min
        )
        return [opt], [sch]

    @staticmethod
    def calc_verocity(x):
        v = torch.diff(x, dim=1)
        return v

    def training_step(self, batch, batch_idx):
        x, seq_lens, labels = batch
        x = x * self.mag

        v = self.calc_verocity(x)

        t, vt, ut, dt, v0, v1 = self.cfm.sample_location(v, seq_lens)

        # calc at
        at = self.net(t, vt, labels)

        # calc loss
        loss_a = F.mse_loss(at, ut, reduction="none")
        loss_a = loss_a.sum(dim=-1).mean()

        loss_v0 = F.mse_loss(vt - at * dt, v0, reduction="none")
        loss_v0 = loss_v0.sum(dim=-1).mean()
        loss_v1 = F.mse_loss(vt + at * (1 - dt), v1, reduction="none")
        loss_v1 = loss_v1.sum(dim=-1).mean()
        loss_v = loss_v0 + loss_v1

        loss = loss_a + loss_v
        loss_dict = dict(a=loss_a, v=loss_v, loss=loss)
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, steps=None):
        x, seq_lens, labels, x_min, x_max = batch
        x_min = x_min.cpu().numpy().reshape(x_min.size(-1))
        x_max = x_max.cpu().numpy().reshape(x_max.size(-1))
        x = x * self.mag
        v = self.calc_verocity(x)
        b, _, pt, d = x.size()

        self.steps = steps

        results = []
        for i in range(b):
            n_ode = NeuralODE(
                wrapper(self.net, labels[i]),
                "dopri5",
                atol=1e-8,
                rtol=1e-8,
                sensitivity="adjoint",
            )

            xi = x[i, 1 : seq_lens[i]]  # remove padding
            vi = v[i, : seq_lens[i] - 1]  # remove padding
            xit = xi[0]
            vit = vi[0]

            vi_preds = []
            xi_preds = []
            pred_len = seq_lens[i] - 1
            for t in range(pred_len):
                xit = xi[t]
                vit = vi[t]

                # update vt
                vit = vit.view(1, pt, d)
                vit = n_ode.trajectory(
                    vit, t_span=torch.linspace(t, t + 1, self.steps + 1)
                )
                vit = vit.view(self.steps + 1, pt, d)

                # test plot
                if t % 5 == 0:
                    plot_traj(vit, vi, t)

                vit = vit[-1]
                vi_preds.append(vit)

                # # update xt
                xit = xit.view(1, pt, d)
                xit = xit + vit.view(1, pt, d)
                xi_preds.append(xit)

            vi_preds = torch.cat(vi_preds).view(pred_len, pt, d)
            xi_preds = torch.cat(xi_preds).view(pred_len, pt, d)

            # unscaling
            xi_true = xi.detach().cpu().numpy()
            xi_preds = xi_preds.detach().cpu().numpy()
            xi_true = xi_true * (x_max - x_min) + x_min
            xi_preds = xi_preds * (x_max - x_min) + x_min
            results.append(
                {
                    "x_true": xi_true,
                    "v_true": vi.detach().cpu().numpy(),
                    "x_pred": xi_preds,
                    "v_pred": vi_preds.detach().cpu().numpy(),
                }
            )

        return results


class wrapper(nn.Module):
    def __init__(self, model, label):
        super().__init__()
        self.model = model
        self.label = label

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x, self.label)


def plot_traj(vit, vi, t):
    pt, d = vit.shape[-2:]
    vit_plot = vit.detach().cpu().numpy()
    vit_pre = vi[t].detach()
    vit_pre = vit_pre.view(pt, d).cpu().numpy()
    vit_nxt = vi[t + 1].detach()
    vit_nxt = vit_nxt.view(pt, d).cpu().numpy()
    vit_mdl = vit_nxt * 0.5 + vit_pre * 0.5
    plt.scatter(vit_pre[:, 0], vit_pre[:, 1], s=4, c="lime")
    plt.scatter(vit_nxt[:, 0], vit_nxt[:, 1], s=4, c="red")
    plt.scatter(vit_mdl[:, 0], vit_mdl[:, 1], s=4, c="pink")
    plt.scatter(vit_plot[0, :, 0], vit_plot[0, :, 1], s=2, c="black")
    plt.scatter(vit_plot[:, :, 0], vit_plot[:, :, 1], s=1, c="olive")
    plt.scatter(vit_plot[-1, :, 0], vit_plot[-1, :, 1], s=2, c="blue")
    for j in range(0, vit_pre.shape[0]):
        plt.plot(
            [vit_pre[j, 0], vit_nxt[j, 0]],
            [vit_pre[j, 1], vit_nxt[j, 1]],
            c="skyblue",
            linestyle="--",
            linewidth=1,
        )
    # plt.xlim(-0.01, 0.01)
    # plt.ylim(-0.01, 0.01)
    plt.show()
