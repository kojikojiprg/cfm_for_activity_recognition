import torch
import torch.utils


class ConditionalFlowMatcher:
    def __init__(self, config):
        self.tau_steps = config.tau_steps
        self.sigma = config.sigma

    def mu_tau(self, vt_cur, vt_nxt, tau):
        return tau * vt_nxt + (1 - tau) * vt_cur

    def sigma_tau(self, tau):
        return tau * self.sigma

    def sample_vt(self, vt_cur, vt_nxt, tau, eps):
        mu_tau = self.mu_tau(vt_cur, vt_nxt, tau)
        sigma_tau = self.sigma_tau(tau)
        return mu_tau + sigma_tau * eps

    def conditional_flow(self, vt_cur, vt_nxt):
        return (vt_nxt - vt_cur) / self.tau_steps

    def sample_location_and_conditional_flow(self, v, seq_len=None, t=None, tau=None, return_noise=False):
        # v (seq_len, pt, d)
        _, pt, d = v.size()
        if t is None:
            t = torch.randint(high=seq_len, size=(1,))
        if tau is None:
            if self.tau_steps > 1:
                tau = torch.randint(high=self.tau_steps, size=(1,), device=v.device) / self.tau_steps
            else:
                tau = torch.zeros((1,))

        vt_cur, vt_nxt = v[t], v[t + 1]

        eps = torch.randn((pt, d), device=v.device)
        vt_tau = self.sample_vt(vt_cur, vt_nxt, tau, eps)
        ut = self.conditional_flow(vt_cur, vt_nxt)  # (pt, d)

        vt_tau = vt_tau.to(torch.float32)
        ut = ut.to(torch.float32)
        if return_noise:
            return tau, vt_tau, ut, eps
        else:
            return tau, vt_tau, ut
