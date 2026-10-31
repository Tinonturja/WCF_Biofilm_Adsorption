import os
import time
import numpy as np
import torch
import torch.nn as nn

from .wcf_common import RES_DIR   

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

# ======================================================================
# REAL WCF DATA
# ======================================================================
from .data import VM_KIN, VM_ISO, t_kin, q_kin, C0_iso, q_iso
c_kin  = np.full_like(t_kin, 40.0)
vm_kin = np.full_like(t_kin, VM_KIN)

c_iso  = C0_iso
t_iso  = np.full_like(c_iso, 170.0)
vm_iso = np.full_like(c_iso, VM_ISO)

T_ALL  = np.concatenate([t_kin,  t_iso])
C_ALL  = np.concatenate([c_kin,  c_iso])
VM_ALL = np.concatenate([vm_kin, vm_iso])
Q_ALL  = np.concatenate([q_kin,  q_iso])


# ======================================================================
# MODEL
# ======================================================================
class SurfacePINN(nn.Module):
    def __init__(self, n_hidden=32, n_layers=3,
                 lam_pso=1.0, lam_iso=1.0, lam_ic=1.0,
                 lr=3e-3, epochs=15000, n_coll=12, seed=0,
                 scaling="nondim", ckpt_path=None, t_scale=200.0):
        super().__init__()
        # scaling="raw"   : loss terms in physical units, as in the original submission
        #                   (physics terms effectively ~5e3x under-weighted vs. the data term).
        # scaling="nondim": every residual divided by its natural scale (q by Q = max q,
        #                   dq/dt by Q/T) so that the lam_* weights are true relative weights.
        if scaling not in ("raw", "nondim"):
            raise ValueError("scaling must be 'raw' or 'nondim'")
        self.scaling, self.ckpt_path, self.T_scale = scaling, ckpt_path, float(t_scale)
        torch.manual_seed(seed)
        self.lam_pso, self.lam_iso, self.lam_ic = lam_pso, lam_iso, lam_ic
        self.lr, self.epochs, self.n_coll = lr, epochs, n_coll

        layers, d_in = [], 3
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_in, n_hidden), nn.Tanh()]
            d_in = n_hidden
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # trainable physics (softplus -> strictly positive)
        self.k2_raw    = nn.Parameter(torch.tensor(-3.7))   
        self.Kf_raw    = nn.Parameter(torch.tensor(-4.4))   
        self.n_inv_raw = nn.Parameter(torch.tensor(0.45))   

    # ---------- scaling ----------
    def _set_scalers(self, t, c, vm, q):
        self.t_m, self.t_s = t.mean(), t.std().clamp_min(1e-8)
        self.c_m, self.c_s = c.mean(), c.std().clamp_min(1e-8)
        self.v_m, self.v_s = vm.mean(), vm.std().clamp_min(1e-8)
        self.q_m, self.q_s = q.mean(), q.std().clamp_min(1e-8)
        self.Q_scale = float(q.abs().max())

    def _forward_real(self, t, c, vm):
        x = torch.cat([(t - self.t_m) / self.t_s,
                       (c - self.c_m) / self.c_s,
                       (vm - self.v_m) / self.v_s], dim=1)
        return self.net(x) * self.q_s + self.q_m

    # ---------- physics helpers ----------
    @property
    def k2(self): return nn.functional.softplus(self.k2_raw)
    @property
    def Kf(self): return nn.functional.softplus(self.Kf_raw)
    @property
    def n_inv(self): return nn.functional.softplus(self.n_inv_raw)

    def qe_implicit(self, C0, vm, n_iter=40, damping=0.3):
        Kf, n_inv = self.Kf, self.n_inv
        qe = 0.5 * C0 * vm                      
        for _ in range(n_iter):
            Ce = torch.clamp(C0 - qe / vm, min=1e-8)
            qe_new = Kf * Ce.pow(n_inv)
            qe = (1 - damping) * qe + damping * qe_new
        return qe

    # ---------- training ----------
    def fit(self, t_arr, c_arr, vm_arr, q_arr, verbose=True):
        t  = torch.tensor(t_arr,  dtype=torch.float64).view(-1, 1)
        c  = torch.tensor(c_arr,  dtype=torch.float64).view(-1, 1)
        vm = torch.tensor(vm_arr, dtype=torch.float64).view(-1, 1)
        q  = torch.tensor(q_arr,  dtype=torch.float64).view(-1, 1)
        self._set_scalers(t, c, vm, q)

        # collocation grid
        n = self.n_coll
        tg  = torch.linspace(0.0, 200.0, n)
        cg  = torch.linspace(15.0, 65.0, n)
        vg  = torch.linspace(0.030, 0.120, max(4, n // 3))
        T, C, V = torch.meshgrid(tg, cg, vg, indexing="ij")
        tc  = T.reshape(-1, 1)
        cc  = C.reshape(-1, 1)
        vc  = V.reshape(-1, 1)

        t_eq  = torch.full_like(cc, 400.0)      
        t_zero = torch.zeros_like(cc)

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        hist = []
        start_ep = 0

        ckpt_path = self.ckpt_path or os.path.join(RES_DIR, "pinn_surface.ckpt")
        if self.scaling == "nondim":
            sq, sd = 1.0 / self.Q_scale, self.T_scale / self.Q_scale
        else:
            sq, sd = 1.0, 1.0
        if getattr(self, "resume", True) and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, weights_only=False)
            self.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_ep = ck["epoch"]
            hist = ck["hist"]
            if verbose:
                print(f"  resumed from epoch {start_ep}")

        wall_budget = float(os.environ.get("WCF_WALL_BUDGET", "0"))
        t_start = time.time()
        self.completed = True

        for ep in range(start_ep, self.epochs):
            opt.zero_grad()

            # --- 1. DATA ---
            loss_data = (((self._forward_real(t, c, vm) - q) * sq) ** 2).mean()

            # --- 2. PSO RESIDUAL ---
            tc_req = tc.clone().requires_grad_(True)
            q_c = self._forward_real(tc_req, cc, vc)
            dq_dt = torch.autograd.grad(q_c.sum(), tc_req, create_graph=True)[0]
            qe_c = self.qe_implicit(cc, vc)
            loss_pso = (((dq_dt - self.k2 * (qe_c - q_c) ** 2) * sd) ** 2).mean()

            # --- 3. EQUILIBRIUM CONSISTENCY ---
            q_eq = self._forward_real(t_eq, cc, vc)
            Ce_eq = torch.clamp(cc - q_eq / vc, min=1e-8)
            loss_iso = (((q_eq - self.Kf * Ce_eq.pow(self.n_inv)) * sq) ** 2).mean()

            # --- 4. INITIAL CONDITION q(0)=0 ---
            loss_ic = ((self._forward_real(t_zero, cc, vc) * sq) ** 2).mean()

            # ==================================================================
            # DYNAMIC LOSS ANNEALING (Linear Warmup)
            # Ramps the data weight from 0.0 to 1.0 over the first 5000 epochs.
            # ==================================================================
            alpha_data = min(1.0, ep / 5000.0)

            loss = (alpha_data * loss_data
                    + self.lam_pso * loss_pso
                    + self.lam_iso * loss_iso
                    + self.lam_ic * loss_ic)
            
            loss.backward()
            opt.step()
            sched.step()

            if ep % 500 == 0 or ep == self.epochs - 1:
                hist.append([ep, loss_data.item(), loss_pso.item(), loss_iso.item(), loss_ic.item()])
                if verbose and ep % 2500 == 0:
                    print(f"  ep{ep:6d} | data {loss_data.item():.3e} (w={alpha_data:.2f}) "
                          f"| pso {loss_pso.item():.3e} "
                          f"| iso {loss_iso.item():.3e} "
                          f"| ic {loss_ic.item():.3e}")

            if wall_budget > 0 and (time.time() - t_start) > wall_budget:
                torch.save({"model": self.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "epoch": ep + 1, "hist": hist}, ckpt_path)
                self.completed = False
                if verbose:
                    print(f"  wall budget reached at epoch {ep + 1}; checkpoint saved -> {ckpt_path}")
                return self

        torch.save({"model": self.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": self.epochs, "hist": hist}, ckpt_path)

        self.k2_    = float(self.k2.detach())
        self.Kf_    = float(self.Kf.detach())
        self.n_inv_ = float(self.n_inv.detach())
        self.n_     = 1.0 / self.n_inv_
        self.history = np.array(hist)
        return self

    @torch.no_grad()
    def predict(self, t_arr, c_arr, vm_arr):
        t  = torch.tensor(np.atleast_1d(t_arr),  dtype=torch.float64).view(-1, 1)
        c  = torch.tensor(np.atleast_1d(c_arr),  dtype=torch.float64).view(-1, 1)
        vm = torch.tensor(np.atleast_1d(vm_arr), dtype=torch.float64).view(-1, 1)
        return np.asarray(self._forward_real(t, c, vm).view(-1).tolist())


# ======================================================================
# METRICS
# ======================================================================
def r2(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    ss_res = ((y - yh) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

def mae(y, yh):
    return float(np.abs(np.asarray(y) - np.asarray(yh)).mean())


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    import time

    print("=" * 68)
    print("3-INPUT PHYSICS-INFORMED NEURAL NETWORK  q(t, C0, V/m)")
    print("=" * 68)
    
    # IMPORTANT: Delete the old checkpoint before running so the model trains from scratch 
    # with the new annealing schedule.
    ckpt_path = os.path.join(RES_DIR, "pinn_surface.ckpt")
    # with WCF_WALL_BUDGET set, an existing checkpoint is resumed instead of deleted
    if os.path.exists(ckpt_path) and not os.environ.get("WCF_WALL_BUDGET"):
        os.remove(ckpt_path)
        print("Deleted old checkpoint to enforce fresh training run with loss annealing.")
        
    t0 = time.time()
    model = SurfacePINN(n_hidden=32, n_layers=3, epochs=15000).fit(
        T_ALL, C_ALL, VM_ALL, Q_ALL)
    print(f"\ntraining time this invocation: {time.time() - t0:.1f}s")

    if not getattr(model, "completed", True):
        import sys
        print("Training checkpointed but not finished — re-run to continue.")
        sys.exit(75)

    print("\n--- RECOVERED PHYSICAL CONSTANTS ---")
    print(f"  k2  = {model.k2_:.5f} g mg^-1 min^-1   (classical refit: 0.02521)")
    print(f"  Kf  = {model.Kf_:.5f}                  (classical refit: 0.01240)")
    print(f"  n   = {model.n_:.4f}  (1/n = {model.n_inv_:.4f})  (classical refit n: 0.632)")

    # ---- in-sample fit on each arm (all 14 points were used in training) ----
    p_kin = model.predict(t_kin, c_kin, vm_kin)
    p_iso = model.predict(t_iso, c_iso, vm_iso)
    print("\n--- IN-SAMPLE FIT (training data; see revision/ for LOO-CV) ---")
    print(f"  kinetics arm (n=9, V/m={VM_KIN:.5f}): "
          f"R2 = {r2(q_kin, p_kin):.4f}, MAE = {mae(q_kin, p_kin):.5f}")
    print(f"  isotherm arm (n=5, V/m={VM_ISO:.5f}): "
          f"R2 = {r2(q_iso, p_iso):.4f}, MAE = {mae(q_iso, p_iso):.5f}")
    print(f"  all 14 points               : "
          f"R2 = {r2(Q_ALL, model.predict(T_ALL, C_ALL, VM_ALL)):.4f}")

    print(f"\n{'t':>6} {'C0':>5} {'V/m':>8} {'actual':>9} {'PINN':>9}")
    for a, b, c_, d, e in zip(T_ALL, C_ALL, VM_ALL, Q_ALL,
                              model.predict(T_ALL, C_ALL, VM_ALL)):
        print(f"{a:6.0f} {b:5.0f} {c_:8.5f} {d:9.5f} {e:9.5f}")

    # ---- surfaces + curves for the visualiser ----
    tg = np.linspace(0, 180, 70)
    cg = np.linspace(15, 65, 60)
    TT, CC = np.meshgrid(tg, cg, indexing="ij")
    surf_iso = model.predict(TT.ravel(), CC.ravel(),
                             np.full(TT.size, VM_ISO)).reshape(TT.shape)
    surf_kin = model.predict(TT.ravel(), CC.ravel(),
                             np.full(TT.size, VM_KIN)).reshape(TT.shape)

    t_fine = np.linspace(0, 180, 250)
    dosage_curves = {}
    for vm_val in [0.0417, 0.0700, 0.1000, 0.2000]:
        dosage_curves[f"{vm_val:.4f}"] = model.predict(
            t_fine, np.full_like(t_fine, 40.0), np.full_like(t_fine, vm_val))

    c_fine = np.linspace(15, 65, 200)
    iso_curve = model.predict(np.full_like(c_fine, 170.0), c_fine,
                              np.full_like(c_fine, VM_ISO))
    kin_curve = model.predict(t_fine, np.full_like(t_fine, 40.0),
                              np.full_like(t_fine, VM_KIN))

    np.savez(f"{RES_DIR}/pinn_surface_results.npz",
             t_grid=tg, c_grid=cg, surf_iso=surf_iso, surf_kin=surf_kin,
             t_fine=t_fine, c_fine=c_fine,
             iso_curve=iso_curve, kin_curve=kin_curve,
             dosage_keys=np.array(list(dosage_curves.keys())),
             dosage_vals=np.array(list(dosage_curves.values())),
             t_kin=t_kin, q_kin=q_kin, c_iso=c_iso, q_iso=q_iso,
             pred_kin=p_kin, pred_iso=p_iso,
             history=model.history,
             constants=np.array([model.k2_, model.Kf_, model.n_, model.n_inv_]),
             vm_kin=VM_KIN, vm_iso=VM_ISO)
    print(f"\nSaved -> {RES_DIR}/pinn_surface_results.npz  (input for 5_visualize_pinn.py)")