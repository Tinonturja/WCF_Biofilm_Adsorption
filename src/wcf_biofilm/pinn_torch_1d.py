import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class PSO_PINN(nn.Module):
    """Physics-informed NN constrained by the pseudo-second-order rate law."""

    def __init__(self, n_hidden=16, lam=1.0, lr=2e-3, epochs=8000, n_coll=40, seed=0):
        super().__init__()
        self.lam, self.lr, self.epochs, self.n_coll, self.seed = lam, lr, epochs, n_coll, seed
        torch.manual_seed(seed)   # reproducible weight initialisation
        self.net = nn.Sequential(nn.Linear(1, n_hidden), nn.Tanh(), nn.Linear(n_hidden, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # trainable physics parameters (raw -> softplus -> positive value)
        self.qe_raw = nn.Parameter(torch.tensor(0.0))   # softplus(0)  ~ 0.69 mg/g
        self.k2_raw = nn.Parameter(torch.tensor(-3.0))  # softplus(-3) ~ 0.049

    # ---- input/output normalisation transforms (fit on the training data) ----
    def _set_scalers(self, t, y):
        self.t_mean, self.t_std = float(t.mean()), float(t.std(unbiased=False))
        self.y_mean, self.y_std = float(y.mean()), float(y.std(unbiased=False))
        self.t_min, self.t_max = float(t.min()), float(t.max())

    def _tn(self, t):   # normalise contact time
        return (t - self.t_mean) / self.t_std

    def _forward_real(self, t_real):
        """t (real units) -> qt (real units), keeping the graph for autograd."""
        tn = self._tn(t_real)
        yn = self.net(tn)
        return yn * self.y_std + self.y_mean

    # ---------------------------------------------------------------
    def fit(self, t_list, y_list, verbose=False):
        t = torch.tensor([float(v) for v in t_list]).view(-1, 1)
        y = torch.tensor([float(v) for v in y_list]).view(-1, 1)
        self._set_scalers(t, y)
        yn = (y - self.y_mean) / self.y_std                      # normalised targets
        tc = torch.linspace(self.t_min, self.t_max, self.n_coll).view(-1, 1)  # collocation
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sp = nn.functional.softplus
        for ep in range(self.epochs):
            opt.zero_grad()
            # ---- DATA LOSS  (normalised scale) ----
            yhat_n = self.net(self._tn(t))
            data_loss = ((yhat_n - yn) ** 2).mean()
            # ---- PHYSICS LOSS  (real units, autograd derivative) ----
            tc_req = tc.clone().requires_grad_(True)
            qt_c = self._forward_real(tc_req)                    # qt(real)
            dqt_dt = torch.autograd.grad(qt_c.sum(), tc_req, create_graph=True)[0]
            qe, k2 = sp(self.qe_raw), sp(self.k2_raw)
            pso_res = dqt_dt - k2 * (qe - qt_c) ** 2             # PSO residual
            phys_loss = (pso_res ** 2).mean()
            # ---- TOTAL ----
            loss = data_loss + self.lam * phys_loss
            loss.backward()
            opt.step()
            if verbose and ep % 2000 == 0:
                print(f"  ep{ep:5d}  data={data_loss.item():.4e} phys={phys_loss.item():.4e}")
        self.qe_ = float(sp(self.qe_raw))
        self.k2_ = float(sp(self.k2_raw))
        return self

    @torch.no_grad()
    def predict(self, t_list):
        t = torch.tensor([float(x) for x in t_list]).view(-1, 1)
        return self._forward_real(t).view(-1).tolist()


if __name__ == "__main__":
    import time
    t = [0, 10, 20, 30, 50, 80, 110, 140, 170]
    q = [0.0, 0.16306, 0.31258, 0.39029, 0.4958, 0.61295, 0.6918, 0.72048, 0.73692]
    t0 = time.time()
    m = PSO_PINN(epochs=8000).fit(t, q, verbose=True)
    print(f"one full physics-informed fit: {time.time()-t0:.2f}s")
    print(f"recovered qe={m.qe_:.4f} mg/g, k2={m.k2_:.5f} g/mg/min")
