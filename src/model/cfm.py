import torch
import torch.utils


class ConditionalFlowMatcher:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.steps = config.steps
        self.sigma = config.sigma

    def sample_vt(self, v0, v1, dt):
        vt_dt = dt * v1 + (1 - dt) * v0
        eps = torch.randn_like(vt_dt)
        return vt_dt.to(torch.float32) + self.sigma * (1 - dt) * eps

    def sample_ut(self, v0, v1):
        return v1 - v0

    @torch.no_grad()
    def sample_location(self, v0, v1):
        # v (b, seq_len - 1, pt, d)
        b, seq_len, pt, d = v0.size()
        dt = torch.randint(self.steps, (b,)) / self.steps
        dt = dt.view(b, 1, 1, 1).repeat(1, seq_len, pt, d).to(v0.device)

        vt = self.sample_vt(v0, v1, dt)  # dt * v1 + (1 - dt) * v0
        ut = self.sample_ut(v0, v1)

        return dt, vt, ut
