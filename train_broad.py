"""Broadly trained discriminative receiver (reviewer control R12, decision D2).

Same architecture, loss, optimizer and step budget as DiscNRX (train.py), but the
training channels are randomised in the style of DeepRx-type receivers:

  CDL model      uniform over {A, B, C, D, E}          (one model per batch)
  delay spread   log-uniform over [30, 1000] ns        (one value per batch)
  UE speed       uniform over [0, 120] km/h            (per slot, Sionna CDL)
  SNR            uniform over [-5, 25] dB              (per slot, as for DiscNRX)

Under this distribution the delay-spread and CDL-model shift conditions of the
paper (DS200-DS1000, CDL A/B/D/E, S2) and V30/V90 are in-distribution; V150-V400
and the ray-traced S4 set remain outside it.

The delay spread is changed per batch by rescaling the normalised CDL path delays
(Sionna 2.x stores them as `cdl._delays = normalised_delays * delay_spread`).

Resumable exactly like train.py: full state every `checkpoint_every` steps.
Output: results/broad/checkpoints/discnrx_{state,final}.pt,
        results/broad/discnrx_training_log.csv
Usage : python train_broad.py [train.max_steps=50000]
"""
from __future__ import annotations

import csv
import math
import os

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from train import _save_state, _load_state, _quick_bler, _CONFIG_PATH, _ROOT

CDL_MODELS = ("A", "B", "C", "D", "E")
DS_RANGE_NS = (30.0, 1000.0)
SPEED_RANGE_KMH = (0.0, 120.0)
REF_DS_NS = 100.0


class BroadChannel:
    """Per-batch random CDL model and delay spread over one SionnaChannel per model."""

    def __init__(self, config, device):
        from src.channel.sionna_channel import SionnaChannel
        cfg = config.copy()
        cfg.channel.ue_speed_kmh_min, cfg.channel.ue_speed_kmh_max = SPEED_RANGE_KMH
        self.chans, self.base_delays = {}, {}
        for m in CDL_MODELS:
            ch = SionnaChannel(cfg, device=device, cdl_model=m, delay_spread_ns=REF_DS_NS)
            self.chans[m] = ch
            self.base_delays[m] = ch.cdl._delays.clone() / (REF_DS_NS * 1e-9)
        self.pilot_grid = self.chans["C"].pilot_grid
        self.device = device

    def generate_batch(self, batch_size, snr_db=None):
        m = CDL_MODELS[int(torch.randint(len(CDL_MODELS), (1,)))]
        lo, hi = (math.log(v) for v in DS_RANGE_NS)
        ds_ns = math.exp(lo + (hi - lo) * float(torch.rand(1)))
        ch = self.chans[m]
        ch.cdl._delays = self.base_delays[m] * (ds_ns * 1e-9)
        return ch.generate_batch(batch_size, snr_db=snr_db)


def main():
    config = OmegaConf.merge(OmegaConf.load(_CONFIG_PATH), OmegaConf.from_cli())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from src.baselines.discriminative_nrx import DiscriminativeNRX

    results_dir = str(config.get("broad_dir", os.path.join(_ROOT, "results", "broad")))
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_path = os.path.join(ckpt_dir, "discnrx_state.pt")
    final_path = os.path.join(ckpt_dir, "discnrx_final.pt")
    log_path = os.path.join(results_dir, "discnrx_training_log.csv")

    ch = BroadChannel(config, device)
    model = DiscriminativeNRX(config, pilot_grid=ch.pilot_grid).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.train.learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config.train.max_steps))

    start = _load_state(state_path, model, optimizer, scheduler, device)
    torch.manual_seed(int(config.train.get("seed", 42)) + 104729 + start)   # fresh data per session
    max_steps = int(config.train.max_steps)
    if start >= max_steps:
        print(f"[DiscNRX-broad] already at {start}/{max_steps} steps - ALL DONE.")
        return
    matched = ch.chans["C"]
    matched.cdl._delays = ch.base_delays["C"] * (REF_DS_NS * 1e-9)
    print(f"[DiscNRX-broad] resuming {start:,}/{max_steps:,} | BLER@15dB (CDL-C 100 ns)="
          f"{_quick_bler(model, matched):.3f}", flush=True)

    log_every = int(config.train.log_every)
    ckpt_every = int(config.train.checkpoint_every)
    new_file = not os.path.exists(log_path)
    model.train()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["step", "ce_loss", "lr"])
        for step in tqdm(range(start + 1, max_steps + 1), desc="DiscNRX-broad (CE)",
                         initial=start, total=max_steps):
            batch = ch.generate_batch(int(config.train.batch_size))
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch.received_grid, batch.bits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            if step % log_every == 0 or step == start + 1:
                writer.writerow([step, loss.item(), scheduler.get_last_lr()[0]]); f.flush()
            if step % ckpt_every == 0:
                _save_state(state_path, final_path, model, optimizer, scheduler, step)
    _save_state(state_path, final_path, model, optimizer, scheduler, max_steps)
    matched.cdl._delays = ch.base_delays["C"] * (REF_DS_NS * 1e-9)
    print(f"[DiscNRX-broad] done: BLER@15dB (CDL-C 100 ns)={_quick_bler(model, matched):.3f}  ** ALL DONE **",
          flush=True)


if __name__ == "__main__":
    main()
