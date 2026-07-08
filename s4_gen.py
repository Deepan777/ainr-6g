"""Generate a set of ray-traced (Sionna RT, Munich) frequency-domain channels
for the S4 evaluation scenario. CPU-only (RT uses the LLVM backend), saves
results/s4_channels.npy of shape (N_CH, N_rx, N_sc), per-realization normalised.
"""
import os, sys
import numpy as np
import sionna.rt as rt
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__))
cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
N_sc = int(cfg.phy.n_subcarriers)
N_rx = int(cfg.phy.n_rx)
SCS = 30e3
FC = float(cfg.channel.carrier_frequency)
N_CH = 200

# Sionna's subcarrier frequencies (baseband), to match the OFDM grid convention.
fsc = np.array(rt.subcarrier_frequencies(N_sc, SCS)).astype(np.float64)  # (N_sc,)

scene = rt.load_scene(rt.scene.munich)
scene.frequency = FC
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=N_rx, vertical_spacing=0.5,
                                horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
import mitsuba as mi
bbox = scene.mi_scene.bbox(); mn, mx = np.array(bbox.min), np.array(bbox.max)
ctr = ((mn + mx) / 2).astype(float)
ps = rt.PathSolver()
gnb = [float(ctr[0]), float(ctr[1]), 40.0]


def rt_hfreq(ue_xy):
    for n in list(scene.transmitters): scene.remove(n)
    for n in list(scene.receivers): scene.remove(n)
    scene.add(rt.Transmitter("ue", position=[float(ue_xy[0]), float(ue_xy[1]), 1.5]))
    scene.add(rt.Receiver("gnb", position=gnb))
    paths = ps(scene, max_depth=4, samples_per_src=300000)
    a, tau = paths.cir()
    a = np.squeeze(np.array(a)); tau = np.squeeze(np.array(tau))
    if a.ndim != 3:   # expect (2, N_rx, P)
        return None
    ac = (a[0] + 1j * a[1]).astype(np.complex128)   # (N_rx, P)
    td = np.atleast_1d(tau).astype(np.float64)       # (P,)
    valid = np.isfinite(td) & (td >= 0)
    if valid.sum() == 0:
        return None
    ac, td = ac[:, valid], td[valid]
    H = ac @ np.exp(-2j * np.pi * np.outer(fsc, td)).T   # (N_rx, N_sc)
    p = np.mean(np.abs(H) ** 2)
    if p < 1e-20:
        return None
    return (H / np.sqrt(p)).astype(np.complex64)


rng = np.random.default_rng(0)
Hs, tries = [], 0
while len(Hs) < N_CH and tries < N_CH * 4:
    tries += 1
    dx, dy = rng.uniform(-200, 200), rng.uniform(-200, 200)
    H = rt_hfreq([ctr[0] + dx, ctr[1] + dy])
    if H is not None:
        Hs.append(H)
        if len(Hs) % 25 == 0:
            print(f"  {len(Hs)}/{N_CH} channels ({tries} tries)", flush=True)

Hs = np.stack(Hs)
np.save(os.path.join(_ROOT, "results", "s4_channels.npy"), Hs)
print(f"Saved {Hs.shape} to results/s4_channels.npy", flush=True)
print("S4GEN_DONE", flush=True)
