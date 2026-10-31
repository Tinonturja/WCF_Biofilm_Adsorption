"""Regression tests: the numbers reported in the manuscript must come out of this code."""
import numpy as np
import pytest

from wcf_biofilm import classical as C
from wcf_biofilm.data import t_kin, q_kin, Ce_iso, q_iso, Q_ALL


def test_data_shape():
    assert len(t_kin) == 9 and len(q_iso) == 5 and len(Q_ALL) == 14


def test_pso_nonlinear_fit():
    (qe, k2), _ = C.fit_kinetic("PSO", t_kin, q_kin)
    assert qe == pytest.approx(0.93601, abs=2e-5)
    assert k2 == pytest.approx(0.02521, abs=2e-5)
    assert C.r2(q_kin, C.pso(t_kin, qe, k2)) == pytest.approx(0.9983, abs=1e-4)


def test_pfo_fit():
    (qe, k1), _ = C.fit_kinetic("PFO", t_kin, q_kin)
    assert C.r2(q_kin, C.pfo(t_kin, qe, k1)) == pytest.approx(0.9964, abs=1e-4)


def test_freundlich_nonlinear_refit():
    (Kf, n), _ = C.fit_isotherm("Freundlich", Ce_iso, q_iso)
    assert Kf == pytest.approx(0.01240, abs=2e-5)
    assert n == pytest.approx(0.6320, abs=2e-4)


def test_gap_percentages():
    qe = 0.93601
    assert 100 * (qe - q_kin[-1]) / qe == pytest.approx(21.3, abs=0.05)       # below the fit
    assert 100 * (qe - q_kin[-1]) / q_kin[-1] == pytest.approx(27.0, abs=0.05)  # above the data


def test_unified_surface_sequential():
    from wcf_biofilm.data import C0_KIN, VM_KIN, C0_iso, VM_ISO
    Kf, n, k2 = 0.01240, 0.6320, 0.02521
    kin = C.surface(t_kin, C0_KIN, VM_KIN, Kf, n, k2)
    iso = np.array([C.solve_qe(c, VM_ISO, Kf, n) for c in C0_iso])
    assert C.r2(q_kin, kin) == pytest.approx(0.962, abs=1e-3)
    assert C.r2(q_iso, iso) == pytest.approx(0.992, abs=1e-3)
