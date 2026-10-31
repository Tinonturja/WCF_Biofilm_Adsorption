"""Physics-informed neural network for the kinetics arm, constrained by the
pseudo-second-order (PSO) rate law  dq/dt = k2 (qe - q)^2.

Two loss formulations are available:

scaling="raw"   The formulation used in the original submission. The data term is
                computed on standardised q, the PSO residual in raw units
                (mg g^-1 min^-1). On a common dimensionless scale the physics term's
                effective weight is lambda * s_q^2 / T^2 (s_q = 0.247 mg/g, T = 180 min),
                i.e. ~1.9e-6 for lambda = 1: the rate law barely constrains the network.
                Kept so that the submitted numbers remain reproducible.

scaling="nondim" (default) Both terms are non-dimensionalised with the same scales,
                u = q/Q and tau = t/T, so that
                    data residual    = (q_hat - q)/Q
                    physics residual = (dq/dt - k2 (qe - q)^2) * T/Q
                and lambda is the actual relative weight. Collocation points cover the
                fixed domain [0, t_domain] rather than only the span of the training
                fold, and q(0) = 0 is imposed exactly through q = Q * tau * N(tau).
"""
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class PSO_PINN(nn.Module):
    def __init__(self, n_hidden=16, lam=1.0, lr=2e-3, epochs=8000, n_coll=40, seed=0,
                 scaling="nondim", hard_ic=None, t_domain=180.0,
                 lam_ramp=None, lbfgs_iters=None):
        super().__init__()
        if scaling not in ("raw", "nondim"):
            raise ValueError("scaling must be 'raw' or 'nondim'")
        self.scaling = scaling
        self.hard_ic = (scaling == "nondim") if hard_ic is None else hard_ic
        # optimisation schedule: physics weight ramped 0 -> lam over `lam_ramp` Adam
        # epochs, then full-batch L-BFGS refinement. Off for the original formulation.
        nd = scaling == "nondim"
        self.lam_ramp = (2000 if nd else 0) if lam_ramp is None else lam_ramp
        self.lbfgs_iters = (500 if nd else 0) if lbfgs_iters is None else lbfgs_iters
        self.lam, self.lr, self.epochs, self.n_coll, self.seed = lam, lr, epochs, n_coll, seed
        self.t_domain = float(t_domain)
        torch.manual_seed(seed)   # reproducible weight initialisation
        self.net = nn.Sequential(nn.Linear(1, n_hidden), nn.Tanh(), nn.Linear(n_hidden, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # trainable physics parameters (raw -> softplus -> positive value)
        self.qe_raw = nn.Parameter(torch.tensor(0.0))   # softplus(0)  ~ 0.69 mg/g
        self.k2_raw = nn.Parameter(torch.tensor(-3.0))  # softplus(-3) ~ 0.049

    # ---------------- scales / forward ----------------
    def _set_scalers(self, t, y):
        self.t_mean, self.t_std = float(t.mean()), float(t.std(unbiased=False))
        self.y_mean, self.y_std = float(y.mean()), float(y.std(unbiased=False))
        self.t_min, self.t_max = float(t.min()), float(t.max())
        self.Q = float(y.abs().max())          # uptake scale (nondim)
        self.T = self.t_domain                 # time scale   (nondim)

    def _tn(self, t):   # normalise contact time (raw formulation)
        return (t - self.t_mean) / self.t_std

    def _forward_real(self, t_real):
        """t (min) -> q (mg/g), differentiable."""
        if self.scaling == "raw":
            return self.net(self._tn(t_real)) * self.y_std + self.y_mean
        tau = t_real / self.T
        out = self.net(2.0 * tau - 1.0)
        if self.hard_ic:
            out = tau * out
        return self.Q * out

    # ---------------- training ----------------
    def fit(self, t_list, y_list, verbose=False):
        t = torch.tensor([float(v) for v in t_list]).view(-1, 1)
        y = torch.tensor([float(v) for v in y_list]).view(-1, 1)
        self._set_scalers(t, y)
        if self.scaling == "raw":
            tc = torch.linspace(self.t_min, self.t_max, self.n_coll).view(-1, 1)
        else:
            tc = torch.linspace(0.0, self.T, self.n_coll).view(-1, 1)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sp = nn.functional.softplus

        def losses():
            qe, k2 = sp(self.qe_raw), sp(self.k2_raw)
            tc_req = tc.clone().requires_grad_(True)
            qt_c = self._forward_real(tc_req)
            dqt_dt = torch.autograd.grad(qt_c.sum(), tc_req, create_graph=True)[0]
            pso_res = dqt_dt - k2 * (qe - qt_c) ** 2
            if self.scaling == "raw":
                yn = (y - self.y_mean) / self.y_std
                data_loss = ((self.net(self._tn(t)) - yn) ** 2).mean()
                phys_loss = (pso_res ** 2).mean()
            else:
                data_loss = (((self._forward_real(t) - y) / self.Q) ** 2).mean()
                phys_loss = ((pso_res * self.T / self.Q) ** 2).mean()
            return data_loss, phys_loss

        self.history = []
        for ep in range(self.epochs):
            opt.zero_grad()
            data_loss, phys_loss = losses()
            w = self.lam * (min(1.0, ep / self.lam_ramp) if self.lam_ramp else 1.0)
            loss = data_loss + w * phys_loss
            loss.backward()
            opt.step()
            if ep % 1000 == 0 or ep == self.epochs - 1:
                self.history.append((ep, data_loss.item(), phys_loss.item()))
                if verbose and ep % 2000 == 0:
                    print(f"  ep{ep:5d}  data={data_loss.item():.4e} phys={phys_loss.item():.4e}")
        if self.lbfgs_iters:
            lb = torch.optim.LBFGS(self.parameters(), lr=1.0, max_iter=self.lbfgs_iters,
                                   tolerance_grad=1e-12, tolerance_change=1e-15,
                                   history_size=50, line_search_fn="strong_wolfe")

            def closure():
                lb.zero_grad()
                d, p = losses()
                L = d + self.lam * p
                L.backward()
                return L
            lb.step(closure)
            d, p = losses()
            self.history.append((self.epochs + self.lbfgs_iters, d.item(), p.item()))
        self.qe_ = float(sp(self.qe_raw))
        self.k2_ = float(sp(self.k2_raw))
        self.final_data_loss_, self.final_phys_loss_ = self.history[-1][1:]
        return self

    @torch.no_grad()
    def predict(self, t_list):
        t = torch.tensor([float(x) for x in t_list]).view(-1, 1)
        return self._forward_real(t).view(-1).tolist()


if __name__ == "__main__":
    import time
    from .data import t_kin, q_kin
    for sc in ("raw", "nondim"):
        t0 = time.time()
        m = PSO_PINN(scaling=sc, epochs=8000).fit(t_kin, q_kin, verbose=True)
        print(f"[{sc}] qe={m.qe_:.4f} mg/g, k2={m.k2_:.5f} g/mg/min  ({time.time()-t0:.1f}s)")
