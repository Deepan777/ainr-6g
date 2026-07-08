"""
train.py — Stage A: offline training of AINR and DiscriminativeNRX.

Session-based, resume-safe. Run the SAME command each time; it auto-resumes from
the last full-state checkpoint, runs one session's worth of steps, saves, and
exits. Repeat until it prints "ALL DONE".

    python train.py                         # one session (config.train.session_steps)
    python train.py train.session_steps=15000   # custom session length
    python train.py train.max_steps=200000      # set the grand total
    python train.py train.use_wandb=true        # also log to Weights & Biases

AINR uses the hybrid objective (supervised coded-bit CE for decoding + VFE for
channel/noise); DiscriminativeNRX uses supervised coded-bit CE. Each model keeps
its own full training state (model + optimizer + LR schedule + step) in
``results/checkpoints/<name>_state.pt`` so sessions chain seamlessly, plus a
model-only ``<name>_final.pt`` for evaluation.
"""

from __future__ import annotations

import csv
import os

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_ROOT, "config", "config.yaml")


# --------------------------------------------------------------------------- #
#  Optional Weights & Biases logging (no-op unless enabled)
# --------------------------------------------------------------------------- #
class _Wandb:
    def __init__(self, config):
        self.enabled = bool(config.train.get("use_wandb", False))
        self._wandb = None
        if self.enabled:
            import wandb
            self._wandb = wandb
            wandb.init(project="ainr-6g",
                       config=OmegaConf.to_container(config, resolve=True))

    def log(self, metrics: dict, step: int) -> None:
        if self.enabled:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled:
            self._wandb.finish()


def _abs(config, key_path: str) -> str:
    p = config
    for k in key_path.split("."):
        p = p[k]
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


# --------------------------------------------------------------------------- #
#  Full-state checkpoint helpers (resume across sessions)
# --------------------------------------------------------------------------- #
def _save_state(state_path, model_path, model, optimizer, scheduler, step) -> None:
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "step": int(step),
    }, state_path)
    torch.save(model.state_dict(), model_path)   # model-only, for evaluate.py


def _load_state(state_path, model, optimizer, scheduler, device) -> int:
    if not os.path.exists(state_path):
        return 0
    ck = torch.load(state_path, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optim"])
    scheduler.load_state_dict(ck["sched"])
    return int(ck.get("step", 0))


@torch.no_grad()
def _quick_bler(model, ch, snr_db=15.0, n_batches=4, batch_size=32) -> float:
    """Fast BLER probe for the progress line."""
    from src.evaluation.metrics import hard_bits, count_block_errors
    was_training = model.training
    model.eval()
    te = tot = 0
    for _ in range(n_batches):
        b = ch.generate_batch(batch_size, snr_db=snr_db)
        llr = model(b.received_grid, record_fe=False) if hasattr(model, "free_energy_series") \
            else model(b.received_grid)
        e, n = count_block_errors(hard_bits(llr), b.bits)
        te += e; tot += n
    if was_training:
        model.train()
    return te / tot if tot else float("nan")


# --------------------------------------------------------------------------- #
#  Stage A.1 — AINR (hybrid: supervised coded-CE + VFE channel/noise)
# --------------------------------------------------------------------------- #
def train_ainr(config, device, wb, session_steps) -> None:
    from src.channel.sionna_channel import SionnaChannel
    from src.ainr import AINR

    results_dir, ckpt_dir = _abs(config, "paths.results_dir"), _abs(config, "paths.checkpoint_dir")
    os.makedirs(results_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    state_path = os.path.join(ckpt_dir, "ainr_state.pt")
    final_path = os.path.join(ckpt_dir, "ainr_final.pt")
    log_path = os.path.join(results_dir, "ainr_training_log.csv")

    ch = SionnaChannel(config, device=device)
    model = AINR(config, ch).to(device)
    optimizer = torch.optim.AdamW(model.posterior.parameters(),
                                  lr=float(config.train.learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.train.max_steps))

    start = _load_state(state_path, model, optimizer, scheduler, device)
    # Seed by resume point so each session draws *fresh* data (not a repeat).
    torch.manual_seed(int(config.train.get("seed", 42)) + start)
    max_steps = int(config.train.max_steps)
    if start >= max_steps:
        print(f"[AINR] already at {start}/{max_steps} steps — ALL DONE.")
        return

    target = min(start + int(session_steps), max_steps)
    bler0 = _quick_bler(model, ch)
    print(f"[AINR] resuming {start:,}/{max_steps:,} ({100*start/max_steps:.0f}%) | "
          f"BLER@15dB={bler0:.3f}  -> this session to {target:,}")

    log_every = int(config.train.log_every)
    ckpt_every = int(config.train.checkpoint_every)
    n_samples = int(config.model.n_vfe_samples)
    warmup = max(1, int(config.train.get("warmup_steps", 1000)))
    noise_freeze = bool(config.train.get("noise_freeze_warmup", True))
    new_file = not os.path.exists(log_path)

    model.train()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["step", "loss", "ce", "kl_channel", "kl_noise",
                             "expected_ll", "lr"])
        for step in tqdm(range(start + 1, target + 1), desc="AINR (hybrid)",
                         initial=start, total=max_steps):
            batch = ch.generate_batch(int(config.train.batch_size))
            sigma_override = (batch.no.sqrt()
                              if (noise_freeze and step <= warmup) else None)
            optimizer.zero_grad(set_to_none=True)
            out = model.hybrid_objective(batch.received_grid, batch.bits,
                                         n_samples=n_samples, sigma_override=sigma_override)
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(model.posterior.parameters(), 1.0)
            optimizer.step(); scheduler.step()

            if step % log_every == 0 or step == start + 1:
                lr = scheduler.get_last_lr()[0]
                writer.writerow([step, out.total.item(), out.ce.item(),
                                 out.kl_channel.item(), out.kl_noise.item(),
                                 out.expected_ll.item(), lr]); f.flush()
                wb.log({"ainr/loss": out.total.item(), "ainr/ce": out.ce.item(),
                        "ainr/kl_channel": out.kl_channel.item(),
                        "ainr/kl_noise": out.kl_noise.item(),
                        "ainr/expected_ll": out.expected_ll.item(), "ainr/lr": lr}, step=step)
            if step % ckpt_every == 0:
                _save_state(state_path, final_path, model, optimizer, scheduler, step)

    _save_state(state_path, final_path, model, optimizer, scheduler, target)
    bler1 = _quick_bler(model, ch)
    done = target >= max_steps
    print(f"[AINR] session done: {target:,}/{max_steps:,} | BLER@15dB {bler0:.3f}->{bler1:.3f}"
          + ("  ** ALL DONE **" if done else "  (run again to continue)"))


# --------------------------------------------------------------------------- #
#  Stage A.2 — DiscriminativeNRX (supervised coded-bit cross-entropy)
# --------------------------------------------------------------------------- #
def train_discriminative_nrx(config, device, wb, session_steps) -> None:
    from src.channel.sionna_channel import SionnaChannel
    from src.baselines.discriminative_nrx import DiscriminativeNRX

    results_dir, ckpt_dir = _abs(config, "paths.results_dir"), _abs(config, "paths.checkpoint_dir")
    os.makedirs(results_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    state_path = os.path.join(ckpt_dir, "discnrx_state.pt")
    final_path = os.path.join(ckpt_dir, "discnrx_final.pt")
    log_path = os.path.join(results_dir, "discnrx_training_log.csv")

    ch = SionnaChannel(config, device=device)
    model = DiscriminativeNRX(config, pilot_grid=ch.pilot_grid).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config.train.learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.train.max_steps))

    start = _load_state(state_path, model, optimizer, scheduler, device)
    torch.manual_seed(int(config.train.get("seed", 42)) + 7919 + start)  # fresh data per session
    max_steps = int(config.train.max_steps)
    if start >= max_steps:
        print(f"[DiscNRX] already at {start}/{max_steps} steps — ALL DONE.")
        return

    target = min(start + int(session_steps), max_steps)
    bler0 = _quick_bler(model, ch)
    print(f"[DiscNRX] resuming {start:,}/{max_steps:,} ({100*start/max_steps:.0f}%) | "
          f"BLER@15dB={bler0:.3f}  -> this session to {target:,}")

    log_every = int(config.train.log_every)
    ckpt_every = int(config.train.checkpoint_every)
    new_file = not os.path.exists(log_path)

    model.train()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["step", "ce_loss", "lr"])
        for step in tqdm(range(start + 1, target + 1), desc="DiscNRX (CE)",
                         initial=start, total=max_steps):
            batch = ch.generate_batch(int(config.train.batch_size))
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch.received_grid, batch.bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            if step % log_every == 0 or step == start + 1:
                lr = scheduler.get_last_lr()[0]
                writer.writerow([step, loss.item(), lr]); f.flush()
                wb.log({"discnrx/ce_loss": loss.item(), "discnrx/lr": lr}, step=step)
            if step % ckpt_every == 0:
                _save_state(state_path, final_path, model, optimizer, scheduler, step)

    _save_state(state_path, final_path, model, optimizer, scheduler, target)
    bler1 = _quick_bler(model, ch)
    done = target >= max_steps
    print(f"[DiscNRX] session done: {target:,}/{max_steps:,} | BLER@15dB {bler0:.3f}->{bler1:.3f}"
          + ("  ** ALL DONE **" if done else "  (run again to continue)"))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    config = OmegaConf.merge(OmegaConf.load(_CONFIG_PATH), OmegaConf.from_cli())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(config.train.get("seed", 42)))
    session_steps = int(config.train.get("session_steps", 25000))

    print("=== Stage A: Offline Training (session-based, resume-safe) ===")
    print(f"Device: {device} | session_steps={session_steps:,} | "
          f"max_steps={int(config.train.max_steps):,}")

    wb = _Wandb(config)
    try:
        train_ainr(config, device, wb, session_steps)
        train_discriminative_nrx(config, device, wb, session_steps)
    finally:
        wb.finish()
    print("Session complete. Re-run `python train.py` to continue, or evaluate when ALL DONE.")
