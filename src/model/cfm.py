import torch
import torch.utils


class ConditionalFlowMatcher:
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.sigma = config.sigma

    def sample_dxt(self, dx, t, eps):
        mu_t = dx[:, t - 1]
        return mu_t + self.sigma * eps

    def conditional_flow(self, dxt_1, dxt):
        return dxt - dxt_1

    def sample_location_and_conditional_flow(self, dx, t=None, return_noise=False):
        # dx (b, seq_len - 1, 17, 2)
        b = dx.size(0)
        if t is None:
            t = torch.randint(1, self.seq_len, b).to(torch.float32)

        eps = torch.randn((b, 17, 2))
        dxt = self.sample_dxt(dx, t, eps)  # (b, 17, 2)
        ut = self.conditional_flow(dx[:, t - 1], dx[:, t])  # (b, 17, 2)

        if return_noise:
            return t, dxt, ut, eps
        else:
            return t, dxt, ut
