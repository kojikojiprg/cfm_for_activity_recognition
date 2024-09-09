import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchdyn.core import NeuralODE

from .cfm import ConditionalFlowMatcher
from .nn import UNet


class FlowMatching(LightningModule):
    def __init__(self, config, n_clusters=120, skel_size=(25, 3), is_pretrain=False):
        super().__init__()
        self.config = config
        # self.seq_len = config.seq_len
        # self.steps = config.steps
        # self.sigma = config.sigma
        self.n_clusters = n_clusters
        self.skel_size = skel_size
        self.is_pretrain = is_pretrain
        self.cfm = None
        self.net = None
        self.net_w = None

    def configure_model(self):
        if self.cfm is None:
            self.cfm = ConditionalFlowMatcher(self.config)
        if self.net is None:
            self.net = UNet(self.config, self.n_clusters, self.skel_size)
            if self.is_pretrain:
                self.net.requires_grad_(False)
        if self.net_w is None:
            self.net_w = UNet(self.config, self.n_clusters, self.skel_size)

    def configure_optimizers(self):
        if not self.is_pretrain:
            opt = torch.optim.Adam(
                list(self.net.parameters()) + list(self.net_w.parameters()),
                self.config.lr,
            )
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, self.config.t_max, self.config.lr_min
            )
            return [opt], [sch]
        else:
            opt = torch.optim.Adam(self.net_w.parameters(), self.config.lr)
            return opt

    @staticmethod
    def calc_verocity(x):
        v = torch.diff(x, dim=1)
        return v

    def training_step(self, batch, batch_idx):
        x, seq_lens, labels = batch

        v = self.calc_verocity(x)

        t, vt, ut, dt, v0, v1 = self.cfm.sample_location(v, seq_lens)

        weights = F.sigmoid(self.net_w(t, vt, labels))
        at = self.net(t, vt, labels) * weights

        weights_true = torch.abs(ut)
        weights_true[weights_true < 0.01] = 0.0
        weights_true[weights_true > 1.0] = 1.0
        loss_w = F.mse_loss(weights, weights_true)
        if self.is_pretrain:
            loss_dict = dict(w=loss_w, loss=loss_w)
            self.log_dict(loss_dict, prog_bar=True, logger=True)
            return loss_w

        loss_a = F.mse_loss(at, ut, reduction="none") * weights_true
        loss_a = loss_a.sum(dim=-1).mean()

        loss_v0 = F.mse_loss(vt - at * dt, v0, reduction="none") * weights_true
        loss_v0 = loss_v0.sum(dim=-1).mean()
        loss_v1 = F.mse_loss(vt + at * (1 - dt), v1, reduction="none") * weights_true
        loss_v1 = loss_v1.sum(dim=-1).mean()
        loss_v = loss_v0 + loss_v1

        # b, pt, d = at.size()
        # at = at.view(b * pt, d)
        # ut = ut.view(b * pt, d)
        # weights_cos = torch.norm(ut, dim=-1)
        # weights_cos[weights_cos < 0.01 * np.sqrt(3)] = 0.0
        # weights_cos[weights_cos > 0.01 * np.sqrt(3)] = 1.0
        # target = torch.ones((at.size(0),)).to(self.device)
        # loss_cos = F.cosine_embedding_loss(at, ut, target, reduction="none")
        # loss_cos = (loss_cos * weights_cos).mean()

        loss = loss_a + loss_v + loss_w  # + loss_cos
        loss_dict = dict(a=loss_a, v=loss_v, w=loss_w, loss=loss)
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, steps=None):
        x, seq_lens, labels = batch
        v = self.calc_verocity(x)
        b, _, pt, d = x.size()

        self.steps = steps

        results = []
        for i in range(b):
            n_ode = NeuralODE(
                wrapper(self.net, self.net_w, labels[i]),
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

            results.append(
                {
                    "x_true": xi.detach().cpu().numpy(),
                    "v_true": vi.detach().cpu().numpy(),
                    "x_pred": xi_preds.detach().cpu().numpy(),
                    "v_pred": vi_preds.detach().cpu().numpy(),
                }
            )

        return results


class wrapper(nn.Module):
    def __init__(self, net, net_w, label):
        super().__init__()
        self.net = net
        self.net_w = net_w
        self.label = label

    def forward(self, t, x, *args, **kwargs):
        weights = F.sigmoid(self.net_w(t, x, self.label))
        return self.net(t, x, self.label) * weights


def plot_traj(vit, vi, t):
    # true
    pt, d = vit.shape[-2:]
    vit_pre = vi[t].detach()
    vit_pre = vit_pre.view(pt, d).cpu().numpy()
    vit_nxt = vi[t + 1].detach()
    vit_nxt = vit_nxt.view(pt, d).cpu().numpy()
    vit_mdl = vit_nxt * 0.5 + vit_pre * 0.5
    plt.scatter(vit_pre[:, 0], vit_pre[:, 1], s=4, c="lime")
    plt.scatter(vit_nxt[:, 0], vit_nxt[:, 1], s=4, c="red")
    plt.scatter(vit_mdl[:, 0], vit_mdl[:, 1], s=4, c="pink")
    for j in range(0, vit_pre.shape[0]):
        plt.plot(
            [vit_pre[j, 0], vit_nxt[j, 0]],
            [vit_pre[j, 1], vit_nxt[j, 1]],
            c="skyblue",
            linestyle="--",
            linewidth=1,
        )

    # pred result
    vit_plot = vit.detach().cpu().numpy()
    plt.scatter(vit_plot[0, :, 0], vit_plot[0, :, 1], s=2, c="black")
    plt.scatter(vit_plot[:, :, 0], vit_plot[:, :, 1], s=1, c="olive")
    plt.scatter(vit_plot[-1, :, 0], vit_plot[-1, :, 1], s=2, c="blue")

    plt.xlabel("v_x")
    plt.ylabel("v_y")
    # plt.xlim(-0.01, 0.01)
    # plt.ylim(-0.01, 0.01)
    plt.show()
