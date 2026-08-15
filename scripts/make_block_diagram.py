"""Methodology block diagram for the WCF ML workflow — muted palette, matplotlib only."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

from wcf_common import save   # shared paths + 300 DPI PNG / vector PDF saver

mpl.rcParams.update({'font.family': 'DejaVu Serif'})

# palette (matches the analysis figures)
INK   = '#2b2b2b'
EDGE  = '#4a4a4a'
C_DATA = '#dce4e7'   # pale slate
C_PREP = '#e8ddcf'   # pale sand
C_MODEL= '#d3ddcb'   # pale sage
C_EVAL = '#e3d3d8'   # pale plum
C_OUT  = '#cfe0e0'   # pale teal
ARROW = '#5f6f75'

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis('off')

def box(x, y, w, h, text, fc, fs=10, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                       linewidth=1.1, edgecolor=EDGE, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, color=INK, zorder=3, fontweight='bold' if bold else 'normal')

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
                        linewidth=1.3, color=ARROW, zorder=1,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# Row headers (stage labels)
stages = [('Input', 5.7), ('Preprocessing', 5.7)]

# 1. Data
box(0.3, 4.4, 2.2, 1.1, 'Experimental data\n(WCF biofilm)\n$q_t$ vs time,  $q_e$ vs $C_e$', C_DATA, fs=9)
# 2. Preprocess
box(3.0, 4.4, 2.2, 1.1, 'Preprocessing\nstandardisation\n(zero mean, unit variance)', C_PREP, fs=9)
# 3. Model bank (central) — title at the top of the box, list beneath it
box(5.7, 3.55, 2.5, 2.6, '', C_MODEL)
ax.text(6.95, 5.85, 'Model bank', ha='center', va='center', fontsize=11, color=INK, fontweight='bold', zorder=3)
ax.text(6.95, 5.40, '— GPR (Matérn 5/2)', ha='center', va='center', fontsize=8, color=INK)
ax.text(6.95, 5.10, '— Polynomial (deg 2, 3)', ha='center', va='center', fontsize=8, color=INK)
ax.text(6.95, 4.80, '— Random forest', ha='center', va='center', fontsize=8, color=INK)
ax.text(6.95, 4.50, '— XGBoost', ha='center', va='center', fontsize=8, color=INK)
ax.text(6.95, 4.20, '— PINN (PSO-constrained)', ha='center', va='center', fontsize=8, color=INK)
ax.text(6.95, 3.85, 'physics prior', ha='center', va='center', fontsize=7.5, color='#8a5a6d', style='italic')

# 4. Evaluation
box(8.7, 4.4, 2.0, 1.1, 'Evaluation\nleave-one-out CV\n(R², MAE)', C_EVAL, fs=9)

# 5. PINN physics box (below model bank)
box(3.0, 1.5, 2.6, 1.4, 'Physics constraint\nPSO rate law\n$dq_t/dt = k_2(q_e-q_t)^2$\nadded to loss', C_PREP, fs=8.5)

# 6. Output (below evaluation)
box(8.4, 1.6, 2.3, 1.3, 'Outputs\nbest model (GPR)\nrecovered $q_e$, $k_2$ (PINN)\nuncertainty bands', C_OUT, fs=8.5)

# arrows main flow
arrow(2.5, 4.95, 3.0, 4.95)
arrow(5.2, 4.95, 5.7, 4.95)
arrow(8.2, 4.95, 8.7, 4.95)
# physics into model bank
arrow(4.3, 2.9, 6.2, 3.75)
# eval to outputs
arrow(9.7, 4.4, 9.55, 2.9)

ax.text(5.5, 0.5, 'Figure. Workflow of the machine-learning analysis applied to the WCF biofilm adsorption data.',
        ha='center', va='center', fontsize=9, color=INK, style='italic')

fig.tight_layout()
save(fig, 'wcf_fig_block_diagram')
plt.close()
print('block diagram saved')
