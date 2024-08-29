import torch
import torch.utils


class ConsistencyFlowMatcher:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.sigma = config.sigma

    def sample_vt(self, vt_cur, vt_nxt, dt):
        xt_dt = dt * vt_nxt + (1 - dt) * vt_cur
        return xt_dt.to(torch.float32)

    @torch.no_grad()
    def sample_location(self, v):
        # v (b, seq_len - 1, pt, d)
        b, _, pt, d = v.size()

        # sample t and dt
        t = torch.randint(high=self.seq_len - 2, size=(1,), device=v.device)
        dt = torch.clamp(torch.rand(size=(1,), device=v.device), max=1 - self.sigma)

        # sample vt
        vt_cur, vt_nxt = v[:, t], v[:, t + 1]
        vt1 = self.sample_vt(vt_cur, vt_nxt, 0)
        vt2 = self.sample_vt(vt_cur, vt_nxt, dt)

        return t, t + dt, vt1.view(b, pt, d), vt2.view(b, pt, d)
