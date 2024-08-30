import torch
import torch.utils


class ConsistencyFlowMatcher:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.sigma = config.sigma

    def sample_vt(self, v0, v1, dt):
        vt_dt = dt * v1 + (1 - dt) * v0
        return vt_dt.to(torch.float32)

    @torch.no_grad()
    def sample_location(self, v0, v1):
        # v (b, seq_len - 1, pt, d)
        dt = torch.clamp(torch.rand_like(v0), max=1 - self.sigma)

        # sample vt
        vt0 = self.sample_vt(v0, v1, 0)  # v0
        vt1 = self.sample_vt(v0, v1, dt)  # dt * v1 + (1 - dt) * v0

        return dt, vt0, vt1
