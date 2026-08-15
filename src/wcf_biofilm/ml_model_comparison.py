import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

from .wcf_common import RES_DIR, save   

rng = np.random.default_rng(42)

# ------------------------------------------------------------------
# REAL WCF DATA
# ------------------------------------------------------------------
t   = np.array([0, 10, 20, 30, 50, 80, 110, 140, 170], dtype=float)   # min
qt  = np.array([0.0, 0.16306, 0.31258, 0.39029, 0.4958,
                0.61295, 0.6918, 0.72048, 0.73692])                    # mg/g

# ==================================================================
# 1. PHYSICS-INFORMED NEURAL NETWORK (PINN) - PyTorch Integration
# ==================================================================
from .pinn_torch_1d import PSO_PINN as PyTorch_PSO_PINN

class PSO_PINN(PyTorch_PSO_PINN):
    def predict(self, t_list):
        preds_list = super().predict(t_list)
        return np.array(preds_list).ravel()

# ==================================================================
# 2. MODEL FACTORY + LOO EVALUATION
# ==================================================================
def gpr_factory():
    kernel = 1.0 * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)
    # Reduced restarts from 15 to 5 to prevent the MacBook from hanging
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                    normalize_y=True, random_state=42)

MODELS = {
    'GPR (Matérn 5/2)':      lambda: gpr_factory(),
    'Polynomial (deg 2)':    lambda: Pipeline([('p', PolynomialFeatures(2)), ('l', LinearRegression())]),
    'Polynomial (deg 3)':    lambda: Pipeline([('p', PolynomialFeatures(3)), ('l', LinearRegression())]),
    'Random Forest':         lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost':               lambda: XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0),
    'PINN (PSO-constrained)':lambda: PSO_PINN(n_hidden=16, lam=1.0, lr=2e-3, epochs=8000, seed=0),
}

def train_score(model_name, x, y):
    m = MODELS[model_name]()
    if model_name.startswith('PINN'):
        m.fit(x, y); return r2_score(y, m.predict(x)), m
    s = StandardScaler(); X = s.fit_transform(x.reshape(-1, 1))
    m.fit(X, y); return r2_score(y, m.predict(X)), (m, s)

def loo_scores(model_name, x, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(x):
        m = MODELS[model_name]()
        if model_name.startswith('PINN'):
            m.fit(x[tr], y[tr])
            preds[te] = m.predict(x[te])
        else:
            s = StandardScaler()
            Xtr = s.fit_transform(x[tr].reshape(-1, 1))
            Xte = s.transform(x[te].reshape(-1, 1))
            m.fit(Xtr, y[tr])
            preds[te] = m.predict(Xte)
    return r2_score(y, preds), mean_absolute_error(y, preds)

# ==================================================================
# 3. RUN — KINETICS
# ==================================================================
rows = []
fitted = {}
print("Starting Model Evaluation... (This will take a moment)")

for name in MODELS:
    print(f"Evaluating {name}...") # Added print statement to track progress
    tr_r2, obj = train_score(name, t, qt)
    loo_r2, loo_mae = loo_scores(name, t, qt)
    rows.append({'Model': name, 'Train_R2': round(tr_r2, 4),
                 'LOO_R2': round(loo_r2, 4), 'LOO_MAE': round(loo_mae, 4)})
    fitted[name] = obj

kin_df = pd.DataFrame(rows)
print("\n=== WCF kinetics — model comparison (LOO-CV) ===")
print(kin_df.to_string(index=False))

pinn = fitted['PINN (PSO-constrained)']
print(f"\nPINN recovered physics parameters: qe = {pinn.qe_:.4f} mg/g, k2 = {pinn.k2_:.5f} g/mg/min")

kin_df.to_csv(f"{RES_DIR}/wcf_model_comparison_kinetics.csv", index=False)

# ==================================================================
# 4. PLOTTING
# ==================================================================
print("Plotting Image Started")
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'DejaVu Serif',
    'axes.edgecolor': '#4a4a4a',
    'axes.labelcolor': '#2b2b2b',
    'xtick.color': '#4a4a4a',
    'ytick.color': '#4a4a4a',
    'text.color': '#2b2b2b',
    'axes.linewidth': 0.9,
})
PAL = {
    'data':   '#2b2b2b', 'gpr':    '#3d6b7d', 'poly2':  '#a6772f',
    'poly3':  '#c99b3f', 'rf':     '#7a8b5a', 'xgb':    '#8a5a6d',
    'pinn':   '#b5613f', 'band':   '#c9d5da', 'grid':   '#e7e7e2',
}

t_fine = np.linspace(0, 170, 200)
print("Setting the Parameters")
# ---- FIGURE 1: Model comparison curves ----
fig1, ax = plt.subplots(figsize=(7, 5))
ax.set_facecolor('#fbfbf9'); ax.grid(color=PAL['grid'], linewidth=0.8, zorder=0)
gpr, sg = fitted['GPR (Matérn 5/2)']
ax.plot(t_fine, gpr.predict(sg.transform(t_fine.reshape(-1,1))), color=PAL['gpr'], lw=2, label='GPR', zorder=4)
mo, sc = fitted['Polynomial (deg 2)']; ax.plot(t_fine, mo.predict(sc.transform(t_fine.reshape(-1,1))), color=PAL['poly2'], lw=1.6, label='Polynomial (deg 2)', zorder=3)
mo, sc = fitted['Polynomial (deg 3)']; ax.plot(t_fine, mo.predict(sc.transform(t_fine.reshape(-1,1))), color=PAL['poly3'], lw=1.6, label='Polynomial (deg 3)', zorder=3)
mo, sc = fitted['Random Forest']; ax.plot(t_fine, mo.predict(sc.transform(t_fine.reshape(-1,1))), color=PAL['rf'], lw=1.4, label='Random Forest', zorder=2)
mo, sc = fitted['XGBoost']; ax.plot(t_fine, mo.predict(sc.transform(t_fine.reshape(-1,1))), color=PAL['xgb'], lw=1.4, label='XGBoost', zorder=2)
ax.plot(t_fine, pinn.predict(t_fine), color=PAL['pinn'], lw=2, ls=(0,(5,1.5)), label='PINN (PSO)', zorder=4)
ax.scatter(t, qt, color=PAL['data'], s=42, zorder=6, edgecolors='white', linewidths=0.9, label='Experimental data')
ax.set_xlabel('Contact time (min)'); ax.set_ylabel('$q_t$ (mg g$^{-1}$)')
ax.set_title('Model comparison on WCF adsorption kinetics', color='#2b2b2b', fontsize=12)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False)
fig1.tight_layout(); save(fig1, 'wcf_fig_model_comparison'); plt.close()
print("Figure 1 done")
# ---- FIGURE 2: Train vs LOO R2 (Violet Palette & Bounding Box) ----
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_facecolor('#fbfbf9')
ax2.grid(axis='y', color=PAL['grid'], linewidth=0.8, zorder=0)

x = np.arange(len(kin_df))
w = 0.4

# Apply the Violet Gradient Palette
color_train = '#9c27b0'  # Vibrant violet 
edge_train  = '#7b1fa2'
color_loo   = '#4a148c'  # Deep violet
edge_loo    = '#311b92'

ax2.bar(x - w/2, kin_df['Train_R2'], w, label='Train R²', color=color_train, edgecolor=edge_train, linewidth=0.6, zorder=3)
ax2.bar(x + w/2, kin_df['LOO_R2'], w, label='LOO R²', color=color_loo, edgecolor=edge_loo, linewidth=0.6, zorder=3)

ax2.set_xticks(x)
ax2.set_xticklabels(kin_df['Model'], rotation=35, ha='right', fontsize=8)
ax2.set_ylabel('R²')

# Increased y-limit to 1.20 to create empty space for the legend
ax2.set_ylim(0, 1.20)

ax2.set_title('Training vs leave-one-out R² (WCF)', color='#2b2b2b', fontsize=12)

# Ensure all 4 spines are explicitly drawn to entrap the figure in a box
for s in ['top', 'bottom', 'left', 'right']:
    ax2.spines[s].set_visible(True)
    ax2.spines[s].set_color('#4a4a4a')
    ax2.spines[s].set_linewidth(0.9)

# Placed legend in the upper center, spread across two columns
ax2.legend(frameon=False, loc='upper center', ncol=2)

fig2.tight_layout()
save(fig2, 'wcf_fig_train_vs_loo')
plt.close()

print("Figure 2 Done")
# ---- FIGURE 3: GPR with uncertainty ----
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.set_facecolor('#fbfbf9'); ax3.grid(color=PAL['grid'], linewidth=0.8, zorder=0)
mu, sd = gpr.predict(sg.transform(t_fine.reshape(-1,1)), return_std=True)
ax3.fill_between(t_fine, mu-2*sd, mu+2*sd, color=PAL['band'], alpha=0.85, label='95% confidence interval', zorder=1)
ax3.plot(t_fine, mu, color=PAL['gpr'], lw=2, label='GPR mean', zorder=4)
ax3.plot(t_fine, pinn.predict(t_fine), color=PAL['pinn'], lw=1.8, ls=(0,(5,1.5)), label='PINN (PSO)', zorder=3)
ax3.scatter(t, qt, color=PAL['data'], s=42, zorder=6, edgecolors='white', linewidths=0.9, label='Experimental data')
ax3.set_xlabel('Contact time (min)'); ax3.set_ylabel('$q_t$ (mg g$^{-1}$)')
ax3.set_title('GPR prediction with uncertainty (WCF)', color='#2b2b2b', fontsize=12)
for s in ['top','right']: ax3.spines[s].set_visible(False)
ax3.legend(fontsize=8, frameon=False)
fig3.tight_layout(); save(fig3, 'wcf_fig_gpr_uncertainty'); plt.close()
print("figure 3 done")

print("\nSaved figures: wcf_fig_model_comparison.png, wcf_fig_train_vs_loo.png, wcf_fig_gpr_uncertainty.png")