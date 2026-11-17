
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21952872.svg)](https://doi.org/10.5281/zenodo.21952872)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# WCF Biofilm Adsorption — Modeling Code

Code accompanying: *"Interfacial Mechanisms of Reactive Dye Adsorption on a
Waste-Cotton Derived PVA/TiO2/Cellulose Biofilm: A Molecular Dynamics and
Physics-Informed Predictive Framework"* (under review at *Chemical Engineering Journal Advances*, manuscript CEJA-D-26-01907).

Full derivation, experimental conditions and model comparison are in the
accompanying paper.

## Key results

| Quantity (kinetics arm, n = 9) | Value |
| :--- | :--- |
| PSO, non-linear least squares: `q_e`, `k_2` | 0.936 ± 0.017 mg/g, 0.0252 ± 0.0018 g·mg⁻¹·min⁻¹ |
| PSO leave-one-out R² / MAE | 0.9969 / 0.0109 mg/g |
| PSO-constrained PINN (corrected, λ = 1, 5 seeds): `q_e`, `k_2` | 0.9361 ± 0.0001 mg/g, 0.02520 ± 0.00001 g·mg⁻¹·min⁻¹ |
| PSO-constrained PINN leave-one-out R² / MAE | 0.9969 / 0.0109 mg/g |
| Model choice, PSO vs PFO (AICc) | −76.5 vs −69.7 (Akaike weight of PSO 0.97) |

| Quantity (both arms, n = 14) | Value |
| :--- | :--- |
| Unified surface, in-sample R² (kinetics / concentration series) | 0.962 / 0.992 |
| Unified surface, leave-one-out R² (sequential and joint routes) | 0.988 |
| 3-input PINN (corrected, 5 seeds): `k_2`, `K_f`, `n` | 0.012–0.015, 0.032–0.037, 0.71–0.73 |
| 3-input PINN leave-one-out R² | 0.989–0.990 |

The concentration-series points are uptakes at 170 min, and the kinetic curve was still
rising at 170 min, so the fitted `q_e` values are extrapolated asymptotes rather than
measured equilibrium values. The time needed to approach equilibrium depends on the
kinetic model (95 % of the asymptote after about 2 h under PFO, about 13 h under PSO);
extended contact-time experiments are future work.

See **Revision (v1.1)** below for what changed relative to the submitted version.

## Installation

Two equivalent ways to install — pick one.

**As an editable package (recommended if you'll import `wcf_biofilm` elsewhere):**
```bash
git clone https://github.com/Tinonturja/WCF_Biofilm_Adsorption.git
cd WCF_Biofilm_Adsorption
pip install -e .
```

**Pinned, reproducible environment (exact versions used to generate every figure and result):**
```bash
git clone https://github.com/Tinonturja/WCF_Biofilm_Adsorption.git
cd WCF_Biofilm_Adsorption
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verified on Python 3.10. `torch` (CPU build) is only required for the PINN
scripts (`pinn_torch_1d.py`, `pinn_surface_3d.py`, `visualize_pinn.py`);
`xgboost` is only required by `ml_model_comparison.py`.

## Structure

```
src/wcf_biofilm/
├── data.py                       # loads data/wcf_adsorption_data.csv
├── classical.py                  # PSO/PFO/Freundlich/Langmuir, mass-balance surface, AICc
├── wcf_common.py                 # shared utilities used across the pipeline
├── pso_classical_fit.py          # pseudo-second-order kinetic fit (classical, non-linear least squares)
├── ml_model_comparison.py        # kinetics benchmark (PSO, PFO, GPR, poly, RF, XGBoost, PINN)
├── concentration_analysis.py     # isotherm-arm and removal-% vs concentration analysis
├── unified_surface_physics.py    # three-parameter Freundlich + PSO mass-balance surface
├── pinn_torch_1d.py              # 1-D PSO-constrained physics-informed neural network
├── pinn_surface_3d.py            # three-input PINN over (t, C0, dosage)
└── visualize_pinn.py             # PINN training/validation figure generation

scripts/
└── make_block_diagram.py         # generates the MD/workflow schematic figure

data/wcf_adsorption_data.csv      # all 14 measurements (single source of truth)
revision/                         # analyses added for the CEJA revision (see below)
tests/                            # regression tests for the reported numbers
figures/                          # figures written by the scripts above
results/                          # model outputs, revision summaries and run logs
```

## Reproducing results

Each module can be run standalone, e.g.:

```bash
python -m wcf_biofilm.pso_classical_fit
python -m wcf_biofilm.pinn_torch_1d
python -m wcf_biofilm.ml_model_comparison
```

Outputs (fitted constants, training curves, predicted surfaces) are written to `results/`.

## Data

All measurements are in `data/wcf_adsorption_data.csv` (kinetics arm, n = 9:
C0 = 40 mg/L, V = 0.05 L, m = 1.2 g; isotherm arm, n = 5: C0 = 20–60 mg/L,
V = 0.05 L, m = 0.5 g, contact time 170 min) and are loaded by every script through
`wcf_biofilm/data.py`. The isotherm-arm uptakes were measured at 170 min, before a
verified plateau, so they are q(170 min) rather than equilibrium q_e.

## Revision (v1.1) — what changed and why

1. **PINN loss scaling (both PINNs).** In the submitted code the data term was on
   standardised q and the PSO residual in raw units (mg g⁻¹ min⁻¹), so the physics term
   carried about 1/530,000 (1-D) and 1/40,000 (3-input) of its nominal weight; an ablation shows the submitted 1-D PINN
   gives the same predictions with the physics term switched off. The corrected
   formulation (`scaling="nondim"`, now the default) divides every residual by its
   natural scale (q/Q, dq/dt·T/Q), imposes q(0) = 0 exactly, ramps the physics weight
   and finishes with L-BFGS. The submitted formulation remains available as
   `scaling="raw"` and reproduces the submitted numbers (tests/test_pinn.py).
2. **Benchmark includes the classical equations.** PSO and PFO are evaluated with the
   same leave-one-out protocol and a forward-extrapolation split.
3. **Multi-seed reporting.** Stochastic models are reported over 5 seeds.
4. **Unified surface.** Governing equations and the three fitted parameters (K_f, n, k₂)
   are stated explicitly; a joint fit that models the isotherm arm as q(170 min) is
   compared with the sequential route, with leave-one-out CV and bootstrap intervals.
5. **Single data file, regression tests and CI** that check the reported numbers.

Reproduce the revision analysis:

```bash
cd revision
python benchmark_classical_ml.py          # classical + data-driven models (minutes)
python unified_surface_analysis.py        # unified surface, LOO, bootstrap
# PINN grids: one fit per job (resumable), e.g.
python run_pinn1d_job.py nd_l1 0 loo 3
WCF_WALL_BUDGET=150 python run_pinn3d_job.py nondim 0 full 0
python summarise_pinn1d.py ../results/revision_pinn1d_jobs.jsonl
python summarise_pinn3d.py ../results/revision_pinn3d_jobs.jsonl
python make_revision_figures.py           # figures/rev_fig1..4
pytest -v ../tests
```

### Reproducibility check

Every number in the revision was produced twice, from a clean checkout with the pinned
environment in `requirements.txt`, on two different computers (Apple-silicon macOS and
x86-64 Linux). `revision/compare_reproduction.py` compares a re-run with the committed
results and prints the largest difference per quantity:

```bash
python revision/compare_reproduction.py RERUN_RESULTS_DIR RERUN_REVISION_DIR
```

Classical fits, data-driven models and the unified surface agree to printed precision.
PINN fits agree to within the seed-to-seed spread (for example, 3-input PINN
leave-one-out R² 0.9893/0.9899 vs 0.9894/0.9899 for seeds 0 and 1).

## Versions

| Version | Content |
| :--- | :--- |
| v1.0.0 / v1.0.1 | Code as submitted with the manuscript (v1.0.1 archived on Zenodo) |
| v1.1.0 | Revision for CEJA-D-26-01907: corrected PINN loss scaling, classical baselines, multi-seed and leave-one-out evaluation, joint surface fit, regenerated figures, reproduction check |

The submitted PINN formulation stays available in v1.1.0 as `scaling="raw"`, so both
versions' numbers can be regenerated from the same code.

## Note on `results/`

Model checkpoints (`*.ckpt`) are not version-controlled; rerun the scripts to
regenerate them. Per-job PINN outputs from the revision grids are kept as
`results/revision_pinn*_jobs.jsonl` so every summary number can be traced to a run.

## Citation

If you use this code, please cite:

```bibtex
@article{faisal2026wcf,
  title   = {Interfacial Mechanisms of Reactive Dye Adsorption on a Waste-Cotton
             Derived PVA/TiO2/Cellulose Biofilm: A Molecular Dynamics and
             Physics-Informed Predictive Framework},
  author  = {Faisal, Md Shajjad Khan and Majumder, Tinon Turja and Islam, Kazi Sirajul
             and Mainuddin, Sk. and Hossain, Mohammad Forhad and Sultana, Niger
             and Hasan, Mahmudul},
  journal = {Chemical Engineering Journal Advances},
  year    = {2026},
  note    = {Under review}
}
```

To cite the software itself, use the Zenodo concept DOI, which always resolves to the
latest version: [10.5281/zenodo.21952872](https://doi.org/10.5281/zenodo.21952872).
Each release also has its own version DOI, listed on the Zenodo record.

## License

MIT — see `LICENSE`.
