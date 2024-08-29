import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .cfm import ConsistencyFlowMatcher
from .nn import TransformerEncoder


class ConditionalFlowMatching(LightningModule):
    def __init__(self, config, skel_size=(25, 3)):
        super().__init__()
        self.config = config
        self.tau_steps = self.config.tau_steps
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

    @staticmethod
    def calc_seq_lens(x):
        # calc seq_len of each sample
        seq_lens = []
        for i in range(x.size(0)):
            mask = torch.where(torch.all(~torch.isnan(x[i]), dim=(1, 2)))[0]
            seq_lens.append(len(mask))
        return seq_lens

    def training_step(self, batch, batch_idx):
        x, label = batch
        b, _, pt, d = x.size()

        with torch.no_grad():
            # v = self.calc_verocity(x)
            seq_lens = self.calc_seq_lens(x)

            # sample from cfm
            dt = torch.empty((0,)).to(self.device)
            xt_dt = torch.empty((0, pt, d)).to(self.device)
            for i in range(b):
                x_one = x[i]
                dt1, dt2, xt_dt1, xt_dt2 = self.cfm.sample_location(x_one, seq_lens[i])
                dt = torch.cat([dt, dt1, dt2])
                xt_dt = torch.cat([xt_dt, xt_dt1, xt_dt2], dim=0)

        # calc vt
        dt_int = dt * self.tau_steps  # [0, 1] -> [0, tau_steps]
        vt = self.net(dt_int, xt_dt).view(b, 2, pt, d)

        # calc reconstructed xt
        dt = dt.view(b, 2, 1, 1).repeat(1, 1, pt, d)
        xt_dt = xt_dt.view(b, 2, pt, d)
        xt1 = xt_dt[:, 0] + (1 - dt[:, 0]) * vt[:, 0]
        xt2 = xt_dt[:, 1] + (1 - dt[:, 1]) * vt[:, 1]

        loss = F.mse_loss(xt1, xt2) + F.mse_loss(vt[:, 0], vt[:, 1])
        self.log("loss", loss, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, label = batch
        b, _, pt, d = x.size()
        x = x.to(torch.float32)

        with torch.no_grad():
            v = self.calc_verocity(x)
            seq_lens = self.calc_seq_lens(x)

            # sample from cfm
            results = []
            for i in range(b):
                xt = x[i, 1].view(1, pt, d)
                vt = v[i, 0].view(1, pt, d)
                # vt = v[i, 0]
                vt_lst = []
                xt_lst = []
                for t in range(seq_lens[i]):
                    for dt in range(self.tau_steps):
                        dt = torch.tensor(dt).to(self.device)

                        # update vt
                        vt = self.net(dt, xt)
                        vt_lst.append(vt)

                        # update xt
                        xt = xt + vt / self.tau_steps
                        xt_lst.append(xt)

                # at = torch.cat(at_lst).view(seq_lens[i], self.tau_steps, pt, d)
                vt = torch.cat(vt_lst).view(seq_lens[i], self.tau_steps, pt, d)
                xt = torch.cat(xt_lst).view(seq_lens[i], self.tau_steps, pt, d)

                xb = x[i, 1 : seq_lens[i]]
                vb = v[i, : seq_lens[i] - 1]
                results.append(
                    {
                        "x_true": xb.detach().cpu().numpy(),
                        "v_true": vb.detach().cpu().numpy(),
                        "x_pred": xt.detach().cpu().numpy(),
                        "v_pred": vt.detach().cpu().numpy(),
                        # "a_pred": at.detach().cpu().numpy(),
                    }
                )

        return results
