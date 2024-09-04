import torch
import torch.utils


class ConditionalFlowMatcher:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.steps = config.steps
        self.sigma = config.sigma

    def mu_t(self, v0, v1, t):
        return t * v1 + (1 - t) * v0

    def sigma_t(self, t):
        return self.sigma * (1 - t)

    def sample_vt(self, v0, v1, t):
        b, seq_len, pt, d = v0.size()
        t = t.view(b, 1, 1, 1).repeat(1, seq_len, pt, d)

        mu = self.mu_t(v0, v1, t)
        sigma = self.sigma_t(t)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def sample_ut(self, v0, v1):
        return v1 - v0

    @torch.no_grad()
    def sample_location(self, v0, v1):
        # v (b, seq_len - 1, pt, d)
        if self.steps > 1:
            t = torch.randint(self.steps, (v0.size(0),)) / self.steps
        else:
            t = torch.zeros((v0.size(0),))
        t = t.to(v0.device)

        vt = self.sample_vt(v0, v1, t)  # v0
        ut = self.sample_ut(v0, v1)  # v1 - v0

        return t, vt, ut
