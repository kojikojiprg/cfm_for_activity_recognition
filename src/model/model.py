import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchdyn.core import NeuralODE

from .cfm import ConditionalFlowMatcher
from .nn import TransformerEncoder


class ConditionalFlowMatching(LightningModule):
    def __init__(self, config, skel_size=(25, 3), is_train=True):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.steps = config.steps
        self.sigma = config.sigma
        self.skel_size = skel_size
        self.cfm = None
        self.net = None
        self.n_ode = None

    def configure_model(self):
        if self.cfm is None:
            self.cfm = ConditionalFlowMatcher(self.config)
        if self.net is None:
            self.net = TransformerEncoder(self.config, self.skel_size)

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

        v0 = self.calc_verocity(x0)
        v1 = self.calc_verocity(x1)

        t, vt, ut = self.cfm.sample_location(v0, v1)

        # calc at
        at = self.net(t, vt)

        # calc reconstructed vt
        b, seq_len, pt, d = v0.size()
        t = t.view(b, 1, 1, 1).repeat(1, seq_len, pt, d)
        recon_vt = vt + at * (1 - t)

        # calc reconstructed x
        recon_xt = x0[:, 1:] + recon_vt

        # calc loss
        loss_a = F.mse_loss(at, ut)
        loss_v = F.mse_loss(recon_vt, v1)
        loss_x = F.mse_loss(recon_xt, x1[:, 1:])

        loss = loss_a + loss_v + loss_x
        loss_dict = dict(a=loss_a, v=loss_v, x=loss_x, loss=loss)
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        if self.n_ode is None:
            self.n_ode = NeuralODE(
                self.net, "dopri5", atol=1e-4, rtol=1e-4, sensitivity="adjoint"
            )

        x, seq_lens, label = batch
        v = self.calc_verocity(x)
        b, _, pt, d = x.size()

        # sample from cfm
        results = []
        for i in range(x.size(0)):
            # remove padding
            xi = x[i, : seq_lens[i]]
            vi = v[i, : seq_lens[i] - 1]

            vi_preds = []
            xi_preds = []
            pred_len = seq_lens[i] - self.seq_len
            for t in range(pred_len):
                xit = xi[t + 1 : t + self.seq_len]
                vit = vi[t : t + self.seq_len - 1]

                # update vt
                vit = vit.view(1, self.seq_len - 1, pt, d)
                vit = self.n_ode.trajectory(
                    vit, t_span=torch.linspace(0, 1, self.steps)
                )
                vit = vit[-1]
                vi_preds.append(vit)

                # update xt
                xit = xit.view(1, self.seq_len - 1, pt, d)
                xit = xit + vit
                xi_preds.append(xit)

                v_loss = F.mse_loss(
                    vit,
                    vi[t + 1 : t + self.seq_len].view(1, self.seq_len - 1, pt, d),
                )
                x_loss = F.mse_loss(
                    xit,
                    xi[t + 2 : t + self.seq_len + 1].view(1, self.seq_len - 1, pt, d),
                )
                print(v_loss.item(), x_loss.item())

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
