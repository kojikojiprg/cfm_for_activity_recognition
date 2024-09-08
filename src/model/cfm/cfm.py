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
        return self.sigma

    def sample_vt(self, v0, v1, t):
        mu_t = self.mu_t(v0, v1, t)
        sigma_t = self.sigma_t(t)
        eps = torch.randn_like(mu_t)
        return mu_t + sigma_t * eps

    def sample_ut(self, v0, v1, t):
        return v1 - v0

    def sample_location(self, v0, v1):
        # v (b, seq_len - 1, pt, d)
        if self.steps > 1:
            t = torch.randint(self.steps + 1, (v0.size(0),)) / self.steps
        else:
            t = torch.zeros((v0.size(0),))
        t = t.to(v0.device)
        b, seq_len, pt, d = v0.size()
        t_expand = t.view(b, 1, 1, 1).repeat(1, seq_len, pt, d)

        vt = self.sample_vt(v0, v1, t_expand)
        ut = self.sample_ut(v0, v1, t_expand)

        return t, t_expand, vt, ut


class ConsistencyFlowMatcher:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.steps = config.steps
        self.sigma = config.sigma

    def mu_t(self, v0, v1, t):
        return t * v1 + (1 - t) * v0

    def sigma_t(self, t):
        return self.sigma

    def sample_vt(self, v0, v1, t):
        mu_t = self.mu_t(v0, v1, t)
        sigma_t = self.sigma_t(t)
        eps = torch.randn_like(mu_t)
        return mu_t + sigma_t * eps

    def sample_location(self, v0, v1):
        # v (b, seq_len - 1, pt, d)
        dt0 = torch.zeros((v0.size(0),))
        dt0 = dt0.to(v0.device)
        dt1 = torch.randint(1, self.steps, (v0.size(0),)) / self.steps
        dt1 = dt1.to(v0.device)
        b, seq_len, pt, d = v0.size()
        dt0_expand = dt0.view(b, 1, 1, 1).repeat(1, seq_len, pt, d)
        dt1_expand = dt1.view(b, 1, 1, 1).repeat(1, seq_len, pt, d)

        vdt0 = self.sample_vt(v0, v1, dt0_expand)
        vdt1 = self.sample_vt(v0, v1, dt1_expand)

        return dt0, dt1, dt0_expand, dt1_expand, vdt0, vdt1
