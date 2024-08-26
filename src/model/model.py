import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .cfm import ConditionalFlowMatcher
from .unet import Unet


class ConditionalFlowMatching(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cfm = None
        self.unet = None

    def configure_model(self):
        if self.cfm is None:
            self.cfm = ConditionalFlowMatcher(self.config)
        if self.unet is None:
            self.unet = Unet(self.config)

    def configure_optimizers(self):
        opt = torch.optim.adam.Adam(self.unet.parameters(), self.config.lr)
        return opt

    def calc_dx(self, x):
        dx = torch.diff(x, dim=1)
        return dx

    def training_step(self, batch, batch_idx):
        x = batch
        # x (b, seq_len, 17, 2)

        dx = self.calc_dx(x)
        # dx (b, seq_len - 1, 17, 2)

        t, dxt, ut = self.cfm.sample_location_and_conditional_flow(dx)
        vt = self.unet(t, dxt)

        loss = F.mse_loss(vt - ut)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        # x (b, seq_len, 17, 2)
        dx = self.calc_dx(x)
