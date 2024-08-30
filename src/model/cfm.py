import torch
import torch.utils


class ConditionalFlowMatcher:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.steps = config.steps
        self.sigma = config.sigma

    def mu_dt(self, v0, v1, dt):
        return dt * v1 + (1 - dt) * v0

    def sigma_dt(self, dt):
        return self.sigma * (1 - dt)

    def sample_vt(self, v0, v1, dt):
        b, seq_len, pt, d = v0.size()
        dt = dt.view(b, 1, 1, 1).repeat(1, seq_len, pt, d)

        mu = self.mu_dt(v0, v1, dt)
        sigma = self.sigma_dt(dt)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def sample_ut(self, v0, v1):
        return v1

    @torch.no_grad()
    def sample_location(self, v0, v1):
        # v (b, seq_len - 1, pt, d)
        dt = torch.randint(self.steps, (v0.size(0),)) / self.steps
        dt = dt.to(v0.device)

        vt = self.sample_vt(v0, v1, dt)  # dt * v1 + (1 - dt) * v0
        ut = self.sample_ut(v0, v1)  # v1

        return dt, vt, ut
