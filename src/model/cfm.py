import torch
import torch.utils


class ConditionalFlowMatcher:
    def __init__(self, config):
        # self.seq_len = config.seq_len
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

    def sample_ut(self, v0, v1):
        return v1 - v0

    def sample_location(self, v, seq_lens):
        # v (b, seq_len - 1, pt, d)
        b, _, pt, d = v.size()
        v0 = []
        v1 = []
        t = []
        dt = []
        for i in range(len(seq_lens)):
            ti = torch.randint(seq_lens[i] - 2, (1,))
            dti = torch.randint(self.steps, (1,)) / self.steps
            t.append(ti + dti)
            dt.append(dti)
            v0.append(v[i, ti].view(1, pt, d))
            v1.append(v[i, ti + 1].view(1, pt, d))
        t = torch.cat(t).to(v.device)
        dt = torch.cat(dt).to(v.device)
        v0 = torch.cat(v0, dim=0)
        v1 = torch.cat(v1, dim=0)

        dt_expand = dt.view(b, 1, 1).repeat(1, pt, d)
        vt = self.sample_vt(v0, v1, dt_expand)
        ut = self.sample_ut(v0, v1)

        return t, vt, ut, dt_expand, v0, v1


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
