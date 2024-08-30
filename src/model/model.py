import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .cfm import ConsistencyFlowMatcher
from .nn import TransformerEncoder


class ConditionalFlowMatching(LightningModule):
    def __init__(self, config, skel_size=(25, 3)):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.skel_size = skel_size
        self.cfm = None
        self.net = None

    def configure_model(self):
        if self.cfm is None:
            self.cfm = ConsistencyFlowMatcher(self.config)
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
        b, seq_len, pt, d = x0.size()

        v0 = self.calc_verocity(x0)
        v1 = self.calc_verocity(x1)
        dt, vt0, vt1 = self.cfm.sample_location(v0, v1)

        # calc at
        at0 = self.net(torch.zeros_like(v0), vt0)
        at1 = self.net(dt, vt1)

        # calc reconstructed vt
        vt0 = vt0 + at0
        vt1 = vt1 + (1 - dt) * at1

        # calc reconstructed x
        recon_x1 = x0[:, 1:] + vt0
        recon_x1dt = x0[:, 1:] + vt1

        loss_x = F.mse_loss(recon_x1, recon_x1dt)
        loss_v = F.mse_loss(vt0, vt1)
        loss_a = F.mse_loss(at0, at1)

        loss = loss_x + loss_v + loss_a
        loss_dict = dict(x=loss_x, v=loss_v, a=loss_a, loss=loss)
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, label = batch

        with torch.no_grad():
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
                xit = xi[1 : self.seq_len].view(self.seq_len - 1, pt, d)
                vit = vi[: self.seq_len - 1].view(self.seq_len - 1, pt, d)

                ai_preds = []
                vi_preds = []
                xi_preds = []
                pred_len = xi_seq_len - self.seq_len - 1
                for t in range(pred_len):
                    dt = torch.zeros_like(vit)

                    # pred ait
                    ait = self.net(dt, vit.view(1, self.seq_len - 1, pt, d))
                    ai_preds.append(ait)

                    # update vt
                    vit = vit + ait
                    vi_preds.append(vit)

                    # update xt
                    xit = xit + vit
                    xi_preds.append(xit)

                ait = torch.cat(ai_preds).view(pred_len, self.seq_len - 1, pt, d)
                vit = torch.cat(vi_preds).view(pred_len, self.seq_len - 1, pt, d)
                xit = torch.cat(xi_preds).view(pred_len, self.seq_len - 1, pt, d)
                results.append(
                    {
                        "x_true": xi[self.seq_len :].detach().cpu().numpy(),
                        "v_true": vi[self.seq_len - 1 :].detach().cpu().numpy(),
                        "x_pred": xit.detach().cpu().numpy(),
                        "v_pred": vit.detach().cpu().numpy(),
                        "a_pred": ait.detach().cpu().numpy(),
                    }
                )

        return results
