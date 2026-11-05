"""Unattended driver: dense training, then the snapshot evaluation.

Every stage is resumable, so this script may be re-run after any interruption
(including a power-off) and will continue from where it stopped.

The AINR evaluation runs as soon as the AINR training pass finishes, so the
pre-registered primary analysis is available without waiting for the DiscNRX
control to finish training.
"""
import os, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY_ = os.path.join(ROOT, "venv", "Scripts", "python.exe")


def stage(label, args, tries=3):
    print(f"[all] {label}", flush=True)
    for attempt in range(1, tries + 1):
        r = subprocess.run([PY_] + args, cwd=ROOT)
        print(f"[all] {label} attempt {attempt} exit={r.returncode}", flush=True)
        if r.returncode == 0:
            return True
    return False


stage("stage 1: AINR training, all seeds", ["run_dense.py", "ainr"])
# The pre-registered primary endpoint needs only these, so evaluate them now
# rather than waiting for the DiscNRX control. exp_dense_eval.py skips snapshots
# that do not exist yet and rows already recorded, so it is safe to run twice.
stage("stage 2: evaluation of the AINR snapshots", ["exp_dense_eval.py"])
stage("stage 3: DiscNRX training, all seeds", ["run_dense.py", "discnrx"])
stage("stage 4: evaluation of the DiscNRX snapshots", ["exp_dense_eval.py"])
print("[all] PIPELINE COMPLETE", flush=True)
