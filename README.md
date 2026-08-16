# WCF Biofilm Adsorption — Modeling Code

Code accompanying: *"Interfacial Mechanisms of Reactive Dye Adsorption on a
Waste-Cotton Derived PVA/TiO2/Cellulose Biofilm: A Molecular Dynamics and
Physics-Informed Predictive Framework"* (submitted to *Journal of
Environmental Chemical Engineering*).

## Installation

Two equivalent ways to install — pick one.

**As an editable package (recommended if you'll import `wcf_biofilm` elsewhere):**
```bash
git clone https://github.com/Tinonturja/wcf_biofilm_adsorption.git
cd wcf_biofilm_adsorption
pip install -e .
```

**Pinned, reproducible environment (exact versions used to generate every figure and result):**
```bash
git clone https://github.com/Tinonturja/wcf_biofilm_adsorption.git
cd wcf_biofilm_adsorption
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
├── wcf_common.py                 # shared utilities used across the pipeline
├── pso_classical_fit.py          # pseudo-second-order kinetic fit (classical, non-linear least squares)
├── ml_model_comparison.py        # six-model benchmark on kinetics data (GPR, poly, RF, XGBoost, PINN)
├── concentration_analysis.py     # isotherm-arm and removal-% vs concentration analysis
├── unified_surface_physics.py    # four-parameter Freundlich + PSO mass-balance surface
├── pinn_torch_1d.py              # 1-D PSO-constrained physics-informed neural network
├── pinn_surface_3d.py            # three-input PINN over (t, C0, dosage)
└── visualize_pinn.py             # PINN training/validation figure generation

scripts/
└── make_block_diagram.py         # generates the MD/workflow schematic figure

figures/                          # one script per manuscript figure, regenerates each panel
results/                          # model outputs referenced in the manuscript (see note below)
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

Experimental kinetics (`n=9`) and isotherm (`n=5`) measurements used to train
and evaluate all models are described in the manuscript's Methods section
(Section 3, "Adsorption Properties").

## Note on `results/`

Large binary artifacts (`*.ckpt` model checkpoints) are excluded from version
control via `.gitignore`. If you need the exact trained weights used in the
paper, either regenerate them by running the scripts above, or track them
separately via [Git LFS](https://git-lfs.github.com) if you want them versioned.

## Citation

If you use this code, please cite:

```bibtex
@article{faisal2026wcf,
  title   = {Interfacial Mechanisms of Reactive Dye Adsorption on a Waste-Cotton
             Derived PVA/TiO2/Cellulose Biofilm: A Molecular Dynamics and
             Physics-Informed Predictive Framework},
  author  = {Faisal, Md Shajjad Khan and Majumder, Tinon Turja and Islam, Kazi Sirajul
             and Mainuddin, Sk. and Hossain, Mohammad Forhad and Sultana, Niger},
  journal = {Journal of Environmental Chemical Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see `LICENSE`.
