"""Job list for pipeline.py (re-read every cycle, so steps can be added while it runs).

Each step is a dict:
  name   unique id (also the name of its log file results/exp_logs/pipe_<name>.log)
  lane   "train" or "eval"; one job per lane runs at a time (two jobs share the GPU)
  cmd    argument list, run from the project root with the venv interpreter
  done   callable -> True once the step's final output exists
  ready  callable -> True once its inputs exist (e.g. a training seed has finished)
  sig    substrings identifying the step's process, to adopt a copy already running

Every command below is itself resumable (checkpoints / part files), so a step
that is killed part-way simply continues from its last saved piece when rerun.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OJ = os.path.join(ROOT, "results", "ojcoms")
MAX_STEPS = 50000
SEEDS_NEW = [1, 2, 3, 4]


def _exists(*parts):
    return os.path.exists(os.path.join(ROOT, *parts))


def _last_logged_step(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 400))
            last = f.read().decode(errors="ignore").strip().splitlines()[-1]
        return int(float(last.split(",")[0]))
    except Exception:
        return -1


def _state_step(path):
    try:
        import torch
        return int(torch.load(path, map_location="cpu", weights_only=False).get("step", -1))
    except Exception:
        return -1


def trained(seed):
    """Both models of a training seed have reached MAX_STEPS (checked on the saved state)."""
    d = os.path.join(ROOT, "results", "seeds", f"seed{seed}")
    for name in ("ainr", "discnrx"):
        if _last_logged_step(os.path.join(d, f"{name}_training_log.csv")) < MAX_STEPS:
            return False                      # cheap test first
        if _state_step(os.path.join(d, "checkpoints", f"{name}_state.pt")) < MAX_STEPS:
            return False
    return True


def broad_trained():
    return _state_step(os.path.join(ROOT, "results", "broad", "checkpoints", "discnrx_state.pt")) >= MAX_STEPS


def steps():
    S = []
    # ---- training lane: independent seeds 1-4 (seed 0 = existing checkpoints) ----
    for s in SEEDS_NEW:
        S.append(dict(
            name=f"train_seed{s}", lane="train",
            cmd=["train.py", f"paths.results_dir=results/seeds/seed{s}",
                 f"paths.checkpoint_dir=results/seeds/seed{s}/checkpoints",
                 f"train.seed={s}", f"train.session_steps={MAX_STEPS}",
                 "train.snapshot_steps=[12000,25000]"],
            done=lambda s=s: trained(s), ready=lambda: True,
            sig=["train.py", f"results/seeds/seed{s}"]))
    # Broadly trained DiscNRX (reviewer control R12, decision D2): last in the lane,
    # so it never delays the seed statistics.
    S.append(dict(name="train_broad", lane="train", cmd=["train_broad.py"],
                  done=broad_trained, ready=lambda: True, sig=["train_broad.py"]))

    # ---- evaluation lane, in priority order ----
    def ev(name, cmd, out, ready=lambda: True, sig=None):
        S.append(dict(name=name, lane="eval", cmd=cmd,
                      done=lambda out=out: os.path.exists(os.path.join(OJ, out)),
                      ready=ready, sig=sig or cmd[:1] + cmd[1:3]))

    ev("shift_seed0", ["exp_shift_detect.py", "--seeds", "0"], "shift_slots_seed0.csv",
       sig=["exp_shift_detect.py", "--seeds 0"])
    ev("model_mismatch", ["exp_model_mismatch.py"], "model_mismatch.csv",
       sig=["exp_model_mismatch.py"])
    ev("matched16_seed0", ["exp_matched.py", "--qam", "16", "--seeds", "0"], "matched_qam16_S1_seed0.csv",
       sig=["exp_matched.py", "--qam 16", "--seeds 0"])
    ev("matched64_seed0", ["exp_matched.py", "--qam", "64", "--seeds", "0"], "matched_qam64_S1_seed0.csv",
       sig=["exp_matched.py", "--qam 64"])
    ev("s4_seed0", ["exp_s4.py", "--seeds", "0"], "s4_adapt_seed0.csv", sig=["exp_s4.py", "--seeds 0"])
    for snr in (3, 5):
        ev(f"adapt_S2_snr{snr}_seed0",
           ["exp_adapt.py", "--cond", "S2", "--snr", str(snr), "--seeds", "0"],
           f"adapt_S2_snr{snr}_seed0.csv", sig=["exp_adapt.py", f"--snr {snr}", "--seeds 0"])
    ev("inversion", ["exp_inversion.py"], "inversion_summary.csv", sig=["exp_inversion.py"])
    ev("ablation_seed1", ["exp_ablation.py", "--seed", "1"], "ablation_summary_seed1.csv",
       sig=["exp_ablation.py", "--seed 1"])
    # no_pilot fails completely at seed 1 (BLER 1.0 at every SNR), so it is not
    # repeated; the seeds 2-3 budget goes to the generative-vs-CE backbone test.
    for s in (2, 3):
        ev(f"ablation_seed{s}",
           ["exp_ablation.py", "--seed", str(s), "--variants",
            "full", "no_detach", "pure_vfe"],
           f"ablation_summary_seed{s}.csv", sig=["exp_ablation.py", f"--seed {s}"])
    for s in (2, 3, 4):              # seeds trained with 12k/25k snapshots
        ev(f"budget_seed{s}", ["exp_budget.py", "--seeds", str(s)], f"budget_shift_seed{s}.csv",
           ready=lambda s=s: trained(s), sig=["exp_budget.py", f"--seeds {s}"])
    for s in SEEDS_NEW:
        ev(f"shift_seed{s}", ["exp_shift_detect.py", "--seeds", str(s)],
           f"shift_slots_seed{s}.csv", ready=lambda s=s: trained(s),
           sig=["exp_shift_detect.py", f"--seeds {s}"])
        for snr in (3, 5):
            ev(f"adapt_S2_snr{snr}_seed{s}",
               ["exp_adapt.py", "--cond", "S2", "--snr", str(snr), "--seeds", str(s)],
               f"adapt_S2_snr{snr}_seed{s}.csv", ready=lambda s=s: trained(s),
               sig=["exp_adapt.py", f"--snr {snr}", f"--seeds {s}"])
        ev(f"matched16_seed{s}", ["exp_matched.py", "--qam", "16", "--seeds", str(s)],
           f"matched_qam16_S1_seed{s}.csv", ready=lambda s=s: trained(s),
           sig=["exp_matched.py", "--qam 16", f"--seeds {s}"])
        ev(f"s4_seed{s}", ["exp_s4.py", "--seeds", str(s)], f"s4_adapt_seed{s}.csv",
           ready=lambda s=s: trained(s), sig=["exp_s4.py", f"--seeds {s}"])
    ev("broad", ["exp_broad.py"], "broad_s4_bler.csv", ready=broad_trained, sig=["exp_broad.py"])
    # Timing needs an otherwise idle GPU: only after all training has finished.
    ev("complexity", ["exp_complexity.py"], "complexity.csv",
       ready=lambda: all(trained(s) for s in SEEDS_NEW) and broad_trained(), sig=["exp_complexity.py"])
    return S
