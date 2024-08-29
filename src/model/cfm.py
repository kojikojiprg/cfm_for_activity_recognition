import torch
import torch.utils


class ConsistencyFlowMatcher:
    def __init__(self, config):
        self.tau_steps = config.tau_steps

    def sample_xt_dt(self, xt_cur, xt_nxt, dt):
        xt_dt = dt * xt_nxt + (1 - dt) * xt_cur
        return xt_dt.to(torch.float32)

    def sample_location(self, x, seq_len):
        # x (seq_len, pt, d)
        t = torch.randint(high=seq_len - 1, size=(1,))

        if self.tau_steps > 1:
            dt1 = (
                torch.randint(high=self.tau_steps - 1, size=(1,), device=x.device)
                / self.tau_steps
            )
            dt2 = torch.clamp(
                dt1 + torch.rand(size=(1,), device=x.device),
                max=(1 - 1 / self.tau_steps),
            )
        else:
            dt1 = torch.zeros((1,), device=x.device)
            dt2 = dt1 + torch.rand(size=(1,), device=x.device)

        xt_cur, xt_nxt = x[t], x[t + 1]
        xt_dt1 = self.sample_xt_dt(xt_cur, xt_nxt, dt1)
        xt_dt2 = self.sample_xt_dt(xt_cur, xt_nxt, dt2)

        return dt1, dt2, xt_dt1, xt_dt2
