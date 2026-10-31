"""PINN regression tests (need torch; ~10 s each)."""
import pytest
from wcf_biofilm.data import t_kin, q_kin

torch = pytest.importorskip("torch")


def test_pinn_raw_reproduces_submission():
    from wcf_biofilm.pinn_torch_1d import PSO_PINN
    m = PSO_PINN(seed=0, scaling="raw").fit(t_kin, q_kin)
    assert m.qe_ == pytest.approx(0.9861, abs=5e-4)
    assert m.k2_ == pytest.approx(0.0201, abs=2e-4)


def test_pinn_nondim_recovers_pso():
    from wcf_biofilm.pinn_torch_1d import PSO_PINN
    m = PSO_PINN(seed=0, scaling="nondim", lam=1.0).fit(t_kin, q_kin)
    assert m.qe_ == pytest.approx(0.9360, abs=2e-3)
    assert m.k2_ == pytest.approx(0.0252, abs=3e-4)
