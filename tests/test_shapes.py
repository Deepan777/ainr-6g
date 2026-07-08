"""
tests/test_shapes.py

Shape and sanity checks. Claude Code runs this after each implementation phase.
All tests must pass before moving to the next phase.

Run with: python tests/test_shapes.py
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# --- Make the project root importable and config path absolute, so this test
#     works regardless of the current working directory. ---------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_CONFIG_PATH = os.path.join(_ROOT, "config", "config.yaml")


def run_test(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
        return True
    except NotImplementedError:
        print(f"  [SKIP] {name} — not yet implemented")
        return True
    except Exception as e:
        print(f"  [FAIL] {name} — {e}")
        return False


def main():
    print("=== AINR-6G Shape & Sanity Tests ===\n")
    config = OmegaConf.load(_CONFIG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2  # small batch for shape checks

    all_pass = True

    # ---- Phase 1: SionnaChannel ----
    print("Phase 1 — SionnaChannel")
    def test_channel():
        from src.channel.sionna_channel import SionnaChannel, ChannelBatch
        ch = SionnaChannel(config, device=device)
        batch = ch.generate_batch(batch_size=B, snr_db=10.0)
        assert batch.bits.shape == (B, batch.bits.shape[1]), "bits shape"
        assert batch.received_grid.shape[0] == B, "grid batch dim"
        assert not torch.isnan(batch.received_grid).any(), "NaN in received grid"
    all_pass &= run_test("SionnaChannel.generate_batch()", test_channel)

    # ---- Phase 2: VariationalPosterior ----
    print("\nPhase 2 — VariationalPosterior")
    def test_posterior_forward():
        from src.channel.sionna_channel import SionnaChannel
        from src.variational_posterior import VariationalPosterior
        ch = SionnaChannel(config, device=device)
        batch = ch.generate_batch(batch_size=B, snr_db=10.0)
        Y = batch.received_grid
        post = VariationalPosterior(config).to(device)
        out = post(Y)
        assert out.bit_logits.shape[0] == B, "bit_logits batch"
        assert not torch.isnan(out.bit_logits).any(), "NaN in LLRs"
    all_pass &= run_test("VariationalPosterior.forward()", test_posterior_forward)

    def test_posterior_sample():
        from src.channel.sionna_channel import SionnaChannel
        from src.variational_posterior import VariationalPosterior
        ch = SionnaChannel(config, device=device)
        batch = ch.generate_batch(batch_size=B, snr_db=10.0)
        Y = batch.received_grid
        post = VariationalPosterior(config).to(device)
        samples = post.sample(Y, n_samples=4)
        assert samples.bits_soft.shape[0] == 4, "n_samples dim"
        assert samples.bits_soft.shape[1] == B, "batch dim"
        assert not torch.isnan(samples.bits_soft).any(), "NaN in bit samples"
    all_pass &= run_test("VariationalPosterior.sample()", test_posterior_sample)

    def test_kl_finite():
        from src.channel.sionna_channel import SionnaChannel
        from src.variational_posterior import VariationalPosterior
        ch = SionnaChannel(config, device=device)
        batch = ch.generate_batch(batch_size=B, snr_db=10.0)
        Y = batch.received_grid
        post = VariationalPosterior(config).to(device)
        out = post(Y)
        kl_b = post.kl_bits(out)
        kl_h = post.kl_channel(out)
        kl_n = post.kl_noise(out)
        for v, name in [(kl_b, "kl_bits"), (kl_h, "kl_channel"), (kl_n, "kl_noise")]:
            assert v.ndim == 0, f"{name} must be scalar"
            assert torch.isfinite(v), f"{name} is not finite"
    all_pass &= run_test("VariationalPosterior KL terms finite", test_kl_finite)

    # ---- Phase 3: GenerativeModel ----
    print("\nPhase 3 — GenerativeModel")
    def test_log_prob():
        from src.channel.sionna_channel import SionnaChannel
        from src.variational_posterior import VariationalPosterior
        from src.generative_model import GenerativeModel
        ch = SionnaChannel(config, device=device)
        batch = ch.generate_batch(batch_size=B, snr_db=10.0)
        Y = batch.received_grid
        post = VariationalPosterior(config).to(device)
        gen = GenerativeModel(config, ch).to(device)
        samples = post.sample(Y, n_samples=4)
        lp = gen.log_prob(Y, samples.bits_soft, samples.h_samples, samples.sigma_samples)
        assert lp.shape == (4, B), f"log_prob shape: expected (4, {B}), got {lp.shape}"
        assert torch.isfinite(lp).all(), "log_prob contains inf/nan"
    all_pass &= run_test("GenerativeModel.log_prob()", test_log_prob)

    # ---- Phase 4: VFE ----
    print("\nPhase 4 — VFE")
    def test_vfe_scalar_and_differentiable():
        from src.channel.sionna_channel import SionnaChannel
        from src.variational_posterior import VariationalPosterior
        from src.generative_model import GenerativeModel
        from src.vfe import variational_free_energy
        ch = SionnaChannel(config, device=device)
        batch = ch.generate_batch(batch_size=B, snr_db=10.0)
        Y = batch.received_grid
        post = VariationalPosterior(config).to(device)
        gen = GenerativeModel(config, ch).to(device)
        vfe = variational_free_energy(Y, post, gen, n_samples=4)
        assert vfe.total.ndim == 0, "VFE.total must be scalar"
        assert torch.isfinite(vfe.total), "VFE.total is not finite"
        # Check gradients flow
        vfe.total.backward()
        grad_norm = sum(
            p.grad.norm().item() for p in post.parameters() if p.grad is not None
        )
        assert grad_norm > 0, "No gradients flowing through VFE"
    all_pass &= run_test("VFE scalar and differentiable", test_vfe_scalar_and_differentiable)

    # ---- Phase 5: AINR end-to-end ----
    print("\nPhase 5 — AINR end-to-end")
    def test_ainr_forward():
        from src.channel.sionna_channel import SionnaChannel
        from src.ainr import AINR
        ch = SionnaChannel(config, device=device)
        batch = ch.generate_batch(batch_size=B, snr_db=10.0)
        model = AINR(config, ch).to(device)
        llrs = model(batch.received_grid)
        K = batch.bits.shape[1]
        assert llrs.shape == (B, K), f"LLR shape: expected ({B}, {K}), got {llrs.shape}"
        assert torch.isfinite(llrs).all(), "LLRs contain inf/nan"
        assert len(model.free_energy_series) == 1, "free_energy_series not appended"
    all_pass &= run_test("AINR.forward()", test_ainr_forward)

    # ---- Phase 6: Baselines ----
    print("\nPhase 6 — Baselines")
    def test_baselines_same_interface():
        from src.channel.sionna_channel import SionnaChannel
        from src.ainr import AINR
        from src.baselines.lmmse import LMMSEReceiver
        from src.baselines.discriminative_nrx import DiscriminativeNRX
        ch = SionnaChannel(config, device=device)
        batch = ch.generate_batch(batch_size=B, snr_db=10.0)
        Y, bits = batch.received_grid, batch.bits
        K = bits.shape[1]
        for cls, kwargs in [
            (LMMSEReceiver, {"config": config, "sionna_channel": ch}),
            (DiscriminativeNRX, {"config": config, "pilot_grid": ch.pilot_grid}),
        ]:
            m = cls(**kwargs).to(device)
            llrs = m(Y)
            assert llrs.shape == (B, K), f"{cls.__name__} LLR shape wrong"
    all_pass &= run_test("Baselines have correct interface", test_baselines_same_interface)

    def test_param_count_match():
        from src.channel.sionna_channel import SionnaChannel
        from src.ainr import AINR
        from src.baselines.discriminative_nrx import DiscriminativeNRX
        ch = SionnaChannel(config, device=device)
        ainr = AINR(config, ch)
        disc = DiscriminativeNRX(config, pilot_grid=ch.pilot_grid)
        ratio = ainr.n_parameters / disc.n_parameters
        assert 0.9 <= ratio <= 1.1, (
            f"Parameter count mismatch: AINR={ainr.n_parameters}, "
            f"DiscNRX={disc.n_parameters}, ratio={ratio:.2f}"
        )
    all_pass &= run_test("AINR and DiscriminativeNRX have matched parameter counts", test_param_count_match)

    # ---- Phase 9: Evaluation metrics ----
    print("\nPhase 9 — Evaluation metrics")
    def test_metrics_sanity():
        from src.evaluation.metrics import (
            hard_bits, compute_bler, compute_ber, free_energy_drift_correlation,
        )
        torch.manual_seed(0)
        Bn, K = 64, 1824
        true = (torch.rand(Bn, K) > 0.5).float()

        # Perfect decode -> zero error.
        if compute_bler(true, true) != 0.0:
            raise ValueError("BLER of a perfect decode must be 0.")

        # Random LLRs -> BER ~ 0.5, and (for a 1824-bit block) BLER ~ 1.0.
        rand_llr = torch.randn(Bn, K)
        ber = compute_ber(hard_bits(rand_llr), true)
        bler = compute_bler(hard_bits(rand_llr), true)
        if not (0.45 <= ber <= 0.55):
            raise ValueError(f"random-bit BER should be ~0.5, got {ber:.3f}")
        if bler < 0.99:
            raise ValueError(f"random-bit BLER should be ~1.0, got {bler:.3f}")

        # Free-energy rises at the drift slot -> strong positive correlation.
        fe = [1.0] * 50 + [5.0] * 50
        corr = free_energy_drift_correlation(fe, drift_slot=50)
        if not (corr > 0.9):
            raise ValueError(f"drift correlation should be high, got {corr:.3f}")
    all_pass &= run_test("Metrics: BER~0.5 / BLER~1.0 / drift-corr", test_metrics_sanity)

    def test_scenario_builder():
        from src.evaluation.scenarios import build_scenario_channel
        ch = build_scenario_channel(config, "s2_delay_shift", device)
        if ch.cdl_model != "A":  # S2 is CDL-A
            raise ValueError(f"S2 should map to CDL-A, got CDL-{ch.cdl_model}")
    all_pass &= run_test("Scenario builder maps config -> channel", test_scenario_builder)

    # ---- Summary ----
    print(f"\n{'='*40}")
    if all_pass:
        print("All implemented phases PASSED.")
    else:
        print("Some tests FAILED — fix before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
