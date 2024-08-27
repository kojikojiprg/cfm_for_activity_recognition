import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .cfm import ConditionalFlowMatcher
from .unet import UNet


class ConditionalFlowMatching(LightningModule):
    def __init__(self, config, skel_size=(25, 3)):
        super().__init__()
        self.config = config
        self.tau_steps = self.config.tau_steps
        self.skel_size = skel_size
        self.cfm = None
        self.unet = None

    def configure_model(self):
        if self.cfm is None:
            self.cfm = ConditionalFlowMatcher(self.config)
        if self.unet is None:
            self.unet = UNet(self.config, self.skel_size)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.unet.parameters(), self.config.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, self.config.t_max, self.config.lr_min
        )
        return [opt], [sch]

    @staticmethod
    def calc_verocity(x):
        v = torch.diff(x, dim=1)
        return v

    @staticmethod
    def calc_seq_lens(v):
        # calc seq_len of each sample
        seq_lens = []
        for i in range(v.size(0)):
            mask = torch.where(torch.all(~torch.isnan(v[i]), dim=(1, 2)))[0]
            seq_lens.append(len(mask) - 1)
        return seq_lens

    def training_step(self, batch, batch_idx):
        x, label = batch
        b, _, pt, d = x.size()
        x = x * self.tau_steps

        with torch.no_grad():
            v = self.calc_verocity(x)
            seq_lens = self.calc_seq_lens(v)

            # sample from cfm
            tau = torch.empty((0,)).to(self.device)
            vt_tau = torch.empty((0, pt, d)).to(self.device)
            ut = torch.empty((0, pt, d)).to(self.device)
            for i in range(b):
                v_one = v[i]
                tau_one, vt_tau_one, ut_one = (
                    self.cfm.sample_location_and_conditional_flow(v_one, seq_lens[i])
                )
                tau = torch.cat([tau, tau_one])
                vt_tau = torch.cat([vt_tau, vt_tau_one], dim=0)
                ut = torch.cat([ut, ut_one], dim=0)

        tau = tau * self.tau_steps  # [0, 1] -> [0, tau_steps]
        at = self.unet(tau, vt_tau)

        loss = F.mse_loss(at, ut)
        self.log("loss", loss, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, label = batch
        b, _, pt, d = x.size()
        x = x * self.tau_steps
        x = x.to(torch.float32)

        with torch.no_grad():
            v = self.calc_verocity(x)
            seq_lens = self.calc_seq_lens(v)

            # sample from cfm
            results = []
            for i in range(b):
                xt = x[i, 1]
                vt = v[i, 0]
                at_lst = []
                vt_lst = []
                xt_lst = []
                for t in range(seq_lens[i]):
                    for tau in range(self.tau_steps):
                        tau = torch.tensor(tau).to(self.device)
                        at = self.unet(tau, vt.view(1, pt, d))
                        at_lst.append(at)

                        # update vt
                        vt = vt + (at / self.tau_steps)
                        vt_lst.append(vt)

                        # update xt
                        xt = xt - (vt / self.tau_steps)
                        xt_lst.append(xt)

                at = torch.cat(at_lst).view(seq_lens[i], self.tau_steps, pt, d)
                vt = torch.cat(vt_lst).view(seq_lens[i], self.tau_steps, pt, d)
                xt = torch.cat(xt_lst).view(seq_lens[i], self.tau_steps, pt, d)

                x = x / self.tau_steps
                v = v / self.tau_steps
                xt = xt / self.tau_steps
                vt = vt / self.tau_steps
                at = at / self.tau_steps
                results.append(
                    {
                        "x_true": x[i, : seq_lens[i] + 1].detach().cpu().numpy(),
                        "v_true": v[i, : seq_lens[i]].detach().cpu().numpy(),
                        "x_pred": xt.detach().cpu().numpy(),
                        "v_pred": vt[: seq_lens[i]].detach().cpu().numpy(),
                        "a_pred": at.detach().cpu().numpy(),
                    }
                )

        return results
