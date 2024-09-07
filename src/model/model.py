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
        self.seq_len = config.seq_len
        self.steps = config.steps
        self.sigma = config.sigma
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
        x0, x1, label = batch
        x0 = x0 * self.mag
        x1 = x1 * self.mag

        v0 = self.calc_verocity(x0)
        v1 = self.calc_verocity(x1)

        t, t_exp, vt, ut = self.cfm.sample_location(v0, v1)

        # calc at
        at = self.net(t, vt, label)

        # calc loss
        # weights = F.threshold(1 - torch.abs(ut), 0.01, 0.01)
        weights = 1
        b, seq_len, pt, d = v0.size()
        loss_a = F.mse_loss(at, ut, reduction="none") * weights
        loss_a = loss_a.sum(dim=-1).mean()

        recon_v1 = vt + (at * (1 - t_exp))
        loss_v1 = F.mse_loss(recon_v1, v1, reduction="none") * weights
        loss_v1 = loss_v1.sum(dim=-1).mean()
        recon_v0 = vt - (at * t_exp)
        loss_v0 = F.mse_loss(recon_v0, v0, reduction="none") * weights
        loss_v0 = loss_v0.sum(dim=-1).mean()
        loss_v = loss_v1 + loss_v0

        at = at.view(b * seq_len * pt, d)
        ut = ut.view(b * seq_len * pt, d)
        target = torch.ones((at.size(0),)).to(self.device)
        loss_cos = F.cosine_embedding_loss(at, ut, target)

        loss = loss_a + loss_cos + loss_v
        loss_dict = dict(a=loss_a, cos=loss_cos, v=loss_v, loss=loss)
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, steps=None):
        x, seq_lens, labels = batch
        x = x * self.mag
        v = self.calc_verocity(x)
        b, _, pt, d = x.size()

        self.steps = steps

        # sample from cfm
        results = []
        for i in range(x.size(0)):
            n_ode = NeuralODE(
                wrapper(self.net, labels[i]),
                "dopri5",
                atol=1e-4,
                rtol=1e-4,
                sensitivity="adjoint",
            )

            xi = x[i, : seq_lens[i]]  # remove padding
            vi = v[i, : seq_lens[i] - 1]
            xit = xi[1 : self.seq_len]
            vit = vi[: self.seq_len - 1]

            vi_preds = []
            xi_preds = []
            pred_len = seq_lens[i] - self.seq_len
            for t in range(pred_len):
                # xit = xi[t + 1 : t + self.seq_len]
                # vit = vi[t : t + self.seq_len - 1]

                # update vt
                vit = vit.view(1, self.seq_len - 1, pt, d)
                vit = n_ode.trajectory(vit, t_span=torch.linspace(0, 1, self.steps + 1))
                # vit_preds = [vit.detach()]
                # for dt in range(self.steps):
                #     dt = torch.full((vit.size(0),), dt / self.steps).to(self.device)
                #     ait = self.net(dt, vit, labels[i])
                #     vit = vit + ait / self.steps
                #     vit_preds.append(vit.detach())
                # vit = torch.cat(vit_preds, dim=0)

                # test plot
                n = 25
                vit = vit.view(self.steps + 1, (self.seq_len - 1) * pt, d)
                vit_plot = vit.detach().cpu().numpy()
                vit_pre = vi[t : t + self.seq_len - 1].detach()
                vit_pre = vit_pre.view((self.seq_len - 1) * pt, d).cpu().numpy()
                vit_nxt = vi[t + 1 : t + self.seq_len].detach()
                vit_nxt = vit_nxt.view((self.seq_len - 1) * pt, d).cpu().numpy()
                vit_mdl = vit_nxt * 0.5 + vit_pre * 0.5
                plt.scatter(vit_plot[0, ::n, 0], vit_plot[0, ::n, 1], s=4, c="black")
                plt.scatter(vit_plot[:, ::n, 0], vit_plot[:, ::n, 1], s=1, c="olive")
                plt.scatter(vit_plot[-1, ::n, 0], vit_plot[-1, ::n, 1], s=4, c="blue")
                plt.scatter(vit_pre[::n, 0], vit_pre[::n, 1], s=2, c="lime")
                plt.scatter(vit_nxt[::n, 0], vit_nxt[::n, 1], s=2, c="red")
                plt.scatter(vit_mdl[::n, 0], vit_mdl[::n, 1], s=4, c="pink")
                for j in range(0, vit_pre.shape[0], n):
                    plt.plot(
                        [vit_pre[j, 0], vit_nxt[j, 0]],
                        [vit_pre[j, 1], vit_nxt[j, 1]],
                        c="skyblue",
                        linestyle="--",
                        linewidth=1,
                    )
                plt.scatter(vit_plot[0, 1::n, 0], vit_plot[0, 1::n, 1], s=4, c="black")
                plt.scatter(vit_plot[:, 1::n, 0], vit_plot[:, 1::n, 1], s=1, c="olive")
                plt.scatter(vit_plot[-1, 1::n, 0], vit_plot[-1, 1::n, 1], s=4, c="blue")
                plt.scatter(vit_pre[1::n, 0], vit_pre[1::n, 1], s=2, c="lime")
                plt.scatter(vit_nxt[1::n, 0], vit_nxt[1::n, 1], s=2, c="red")
                plt.scatter(vit_mdl[1::n, 0], vit_mdl[1::n, 1], s=4, c="pink")
                for j in range(1, vit_pre.shape[0], n):
                    plt.plot(
                        [vit_pre[j, 0], vit_nxt[j, 0]],
                        [vit_pre[j, 1], vit_nxt[j, 1]],
                        c="pink",
                        linestyle="--",
                        linewidth=1,
                    )
                # plt.xlim(-0.01, 0.01)
                # plt.ylim(-0.01, 0.01)
                plt.show()

                if t == 1:
                    return []

                vit = vit[-1]
                vi_preds.append(vit)

                # # update xt
                xit = xit.view(1, self.seq_len - 1, pt, d)
                xit = xit + vit.view(1, self.seq_len - 1, pt, d)
                xi_preds.append(xit)

            vi_preds = torch.cat(vi_preds).view(pred_len, self.seq_len - 1, pt, d)
            xi_preds = torch.cat(xi_preds).view(pred_len, self.seq_len - 1, pt, d)
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
    def __init__(self, model, label):
        super().__init__()
        self.model = model
        self.label = label

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x, self.label)
