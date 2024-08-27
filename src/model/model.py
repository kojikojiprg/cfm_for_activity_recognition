import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .cfm import ConditionalFlowMatcher
from .unet import UNet


class ConditionalFlowMatching(LightningModule):
    def __init__(self, config, skel_size=(25, 3)):
        super().__init__()
        self.config = config
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
        return opt

    def calc_verocity(self, x):
        v = torch.diff(x, dim=1)
        return v

    def training_step(self, batch, batch_idx):
        x, label = batch

        v = self.calc_verocity(x)
        mask = torch.all(torch.isnan(v), dim=(2, 3))
        # v (b, seq_len - 1, pt, d)

        b, _, pt, d = x.size()
        tau = torch.empty((0,)).to(self.device)
        vt_tau = torch.empty((0, pt, d)).to(self.device)
        ut = torch.empty((0, pt, d)).to(self.device)
        for i in range(b):
            v_one = v[i][mask[i]]
            tau_one, vt_tau_one, ut_one = self.cfm.sample_location_and_conditional_flow(v_one)
            tau = torch.cat([tau, tau_one])
            vt_tau = torch.cat([vt_tau, vt_tau_one], dim=0)
            ut = torch.cat([ut, ut_one], dim=0)

        tau = tau * self.config.tau_steps  # [0, 1] -> [0, tau_steps]
        vt = self.unet(tau, vt_tau)

        loss = F.mse_loss(vt, ut)
        return loss

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError
