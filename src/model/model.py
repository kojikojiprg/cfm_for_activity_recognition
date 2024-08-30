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

        dt, vt, ut = self.cfm.sample_location(v0, v1)

        # calc at
        at = self.net(dt, vt)

        # calc at0_sum
        at_sum = at[:, 1:].sum(dim=1)
        at_true = v1[:, -1] - v1[:, 0]

        # calc reconstructed vt
        recon_vt = vt + at

        # calc reconstructed x
        recon_xt = x0[:, 1:] + recon_vt

        # calc loss
        loss_a = F.mse_loss(at, ut)
        loss_a_sum = F.mse_loss(at_sum, at_true)
        loss_v = F.mse_loss(recon_vt, v1)
        loss_x = F.mse_loss(recon_xt, x1[:, 1:])

        loss = loss_a + loss_a_sum + loss_v + loss_x
        loss_dict = dict(a=loss_a, a_sum=loss_a_sum, v=loss_v, x=loss_x, loss=loss)
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        if self.n_ode is None:
            self.n_ode = NeuralODE(
                self.net, "dopri5", atol=1e-4, rtol=1e-4, sensitivity="adjoint"
            )

        x, label = batch
        v = self.calc_verocity(x)

        # sample from cfm
        results = []
        for i in range(x.size(0)):
            # remove padding
            mask = ~torch.all(torch.isnan(x[i]), dim=(1, 2))
            xi = x[i][mask]
            vi = v[i][mask[1:]]

            # get init vals
            xi_seq_len, pt, d = xi.size()
            xit = xi[1 : self.seq_len]
            vit = vi[: self.seq_len - 1]

            vi_preds = []
            xi_preds = []
            pred_len = xi_seq_len - self.seq_len
            for t in range(pred_len):
                xit = xi[t + 1 : t + self.seq_len]
                vit = vi[t : t + self.seq_len - 1]
                # vit = vit + 0.1 * torch.randn_like(vit)

                # update vt
                vit = vit.view(1, self.seq_len - 1, pt, d)
                vit = self.n_ode.trajectory(vit, t_span=torch.linspace(0, 1, self.steps))
                # vit (steps, b, seq_len - 1, pt, d)
                vit = vit[-1].view(1, self.seq_len - 1, pt, d)
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
                    "v_loss": v_loss.detach().cpu().numpy(),
                }
            )

        return results


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x)
