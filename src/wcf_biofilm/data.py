"""Single source of truth for the experimental measurements.

All scripts load the data from ``data/wcf_adsorption_data.csv`` through this
module, so the numbers used for every table and figure are defined once.

Kinetics arm : C0 = 40 mg/L, V = 0.05 L, m = 1.2 g  (dosage 24 g/L), t = 0-170 min
Isotherm arm : C0 = 20-60 mg/L, V = 0.05 L, m = 0.5 g (dosage 10 g/L), t = 170 min
Note: the isotherm-arm values are uptakes at t = 170 min. The kinetics arm shows
that uptake is still rising at 170 min, so these are not verified equilibrium
values (q_e); models that need q_e should treat them as q(t = 170 min).
"""
import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "data",
                        "wcf_adsorption_data.csv")


def load():
    return pd.read_csv(DATA_CSV)


_df = load()
_kin = _df[_df.arm == "kinetics"].sort_values("t_min")
_iso = _df[_df.arm == "isotherm"].sort_values("C0_mg_L")

# kinetics arm
t_kin = _kin.t_min.to_numpy(float)
q_kin = _kin.q_mg_g.to_numpy(float)
C0_KIN = float(_kin.C0_mg_L.iloc[0])
VM_KIN = float(_kin.V_L.iloc[0] / _kin.m_g.iloc[0])        # L/g  (0.041667)
DOSE_KIN = 1.0 / VM_KIN                                       # g/L  (24)

# isotherm arm (measured at t = 170 min)
C0_iso = _iso.C0_mg_L.to_numpy(float)
q_iso = _iso.q_mg_g.to_numpy(float)
Ce_iso = _iso.Ce_mg_L.to_numpy(float)
removal_iso = _iso.removal_pct.to_numpy(float)
T_ISO = float(_iso.t_min.iloc[0])
VM_ISO = float(_iso.V_L.iloc[0] / _iso.m_g.iloc[0])          # L/g  (0.1)
DOSE_ISO = 1.0 / VM_ISO                                       # g/L  (10)

# all 14 points as (t, C0, V/m, q)
T_ALL = np.concatenate([t_kin, np.full_like(C0_iso, T_ISO)])
C_ALL = np.concatenate([np.full_like(t_kin, C0_KIN), C0_iso])
VM_ALL = np.concatenate([np.full_like(t_kin, VM_KIN), np.full_like(C0_iso, VM_ISO)])
Q_ALL = np.concatenate([q_kin, q_iso])
ARM_ALL = np.array(["kinetics"] * len(t_kin) + ["isotherm"] * len(C0_iso))
