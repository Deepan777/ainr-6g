"""Dense-snapshot training driver (training-budget stopping-criterion study).

Trains seven independent seeds with frequent model snapshots so the
robustness-vs-budget trajectory can be resolved, instead of the three points
(12k/25k/50k) available from the original runs.

AINR is trained for EVERY seed first, then DiscNRX for every seed. The
pre-registered primary endpoint (ANALYSIS_PROTOCOL_BUDGET.md) uses the AINR
free-energy signal alone, so this ordering completes the paper-critical data in
roughly half the total time; the DiscNRX control then accumulates behind it and
can be interrupted without affecting the primary analysis. Each model keeps its
own resume state, so the split is equivalent to training them together.

Writes ONLY to results/dense/seed<N>/ -- the original results/seeds/ runs are
never touched. Seeds 2-4 repeat the original training seeds, so the snapshots at
12000 and 25000 steps double as a determinism check against the existing
checkpoints; seeds 5-8 are new (held-out).

train.py is resume-safe (it chains from <model>_state.pt), so this driver may be
re-run after any interruption, including a power-off.
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY_ = os.path.join(ROOT, "venv", "Scripts", "python.exe")
SEEDS = [2, 3, 4, 5, 6, 7, 8]   # dev = 2-4 (already seen at 12k/25k/50k), held-out = 5-8
MAX_STEPS = 50000
SNAPS = list(range(1000, 10001, 1000)) + list(range(12500, 50001, 2500))


def done(seed, model):
    d = os.path.join(ROOT, "results", "dense", f"seed{seed}", "checkpoints")
    if not os.path.isdir(d):
        return False
    snaps = [f for f in os.listdir(d) if f.startswith(f"{model}_step")]
    return os.path.exists(os.path.join(d, f"{model}_final.pt")) and len(snaps) >= len(SNAPS)


def run(seed, model):
    if done(seed, model):
        print(f"[dense] seed{seed} {model} already complete, skipping", flush=True)
        return
    for attempt in range(1, 4):
        t0 = time.time()
        print(f"[dense] seed{seed} {model} attempt {attempt} "
              f"starting {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        r = subprocess.run([PY_, "train.py",
                            f"paths.results_dir=results/dense/seed{seed}",
                            f"paths.checkpoint_dir=results/dense/seed{seed}/checkpoints",
                            f"train.seed={seed}",
                            f"train.models={model}",
                            f"train.session_steps={MAX_STEPS}",
                            "train.snapshot_steps=[" + ",".join(map(str, SNAPS)) + "]",
                            # thermal duty cycling: this GPU is clocked from 2100 MHz
                            # down to 210 MHz at 89 C. The pause does not raise
                            # throughput (both settings sustain ~6.4 it/s) but holds
                            # the GPU near 77 C instead of 89 C.
                            "train.cooldown_every=25", "train.cooldown_seconds=2.0"],
                           cwd=ROOT)
        print(f"[dense] seed{seed} {model} attempt {attempt} exit={r.returncode} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        if r.returncode == 0:
            return
        time.sleep(30)
    print(f"[dense] seed{seed} {model} FAILED after 3 attempts -- continuing", flush=True)


if __name__ == "__main__":
    import sys
    # optional argument selects one pass, so the caller can evaluate the AINR
    # snapshots before the DiscNRX control has finished training
    passes = sys.argv[1:] or ["ainr", "discnrx"]
    for model in passes:
        print(f"[dense] ===== pass: {model} =====", flush=True)
        for seed in SEEDS:
            run(seed, model)
    print(f"[dense] PASSES DONE: {','.join(passes)}", flush=True)
