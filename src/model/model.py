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
        x, label = batch
        b, seq_len, pt, d = x.size()

        v = self.calc_verocity(x)
        t1, t2, vt1, vt2 = self.cfm.sample_location(v)

        # calc vt
        at1 = self.net(t1, vt1)
        at2 = self.net(t2, vt2)

        # calc reconstructed xt
        vt1 = vt1 + at1
        vt2 = vt2 + (1 - (t2 - t1)) * at2

        loss = F.mse_loss(vt1, vt2) + F.mse_loss(at1, at2)
        self.log("loss", loss, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, label = batch
        b, _, pt, d = x.size()
        x = x.to(torch.float32)

        with torch.no_grad():
            v = self.calc_verocity(x)

            # sample from cfm
            results = []
            for i in range(b):
                xt = x[i, 1].view(1, pt, d)
                vt = v[i, 0].view(1, pt, d)
                at_lst = []
                vt_lst = []
                xt_lst = []
                for t in range(self.seq_len):
                    # update vt
                    at = self.net(t, vt)
                    at_lst.append(at)

                    # update vt
                    vt = vt + at
                    vt_lst.append(vt)

                    # update xt
                    xt = xt + vt
                    xt_lst.append(xt)

                at = torch.cat(at_lst).view(self.seq_len, pt, d)
                vt = torch.cat(vt_lst).view(self.seq_len, pt, d)
                xt = torch.cat(xt_lst).view(self.seq_len, pt, d)

                xb = x[i, 1 : self.seq_len]
                vb = v[i, : self.seq_len]
                results.append(
                    {
                        "x_true": xb.detach().cpu().numpy(),
                        "v_true": vb.detach().cpu().numpy(),
                        "x_pred": xt.detach().cpu().numpy(),
                        "v_pred": vt.detach().cpu().numpy(),
                        "a_pred": at.detach().cpu().numpy(),
                    }
                )

        return results
