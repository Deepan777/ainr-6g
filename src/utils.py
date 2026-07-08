"""
src/utils.py

Lightweight, dependency-free helpers shared across modules.

The key function is :func:`infer_dims`, which derives the physical-layer and
LDPC code dimensions purely from ``config``. This lets modules that only receive
``config`` (e.g. :class:`VariationalPosterior`, whose constructor takes just the
config) compute the number of information bits ``K`` *consistently* with
:class:`~src.channel.sionna_channel.SionnaChannel`, which obtains the same
numbers from the Sionna resource grid.

Assumption: pilots occupy *entire* OFDM symbols (the indices listed in
``phy.pilot_ofdm_symbol_indices``); all resource elements on the remaining
symbols carry data. This matches the Kronecker pilot configuration used by the
channel wrapper and yields ``num_data_re = N_sc * (N_sym - n_pilot_symbols)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Dims:
    """Derived physical-layer / code dimensions."""

    n_subcarriers: int       # N_sc
    n_symbols: int           # N_sym
    n_rx: int                # gNB antennas
    n_tx: int                # UE antennas / streams
    n_bits_per_symbol: int   # QAM order
    coderate: float          # LDPC rate
    pilot_symbol_indices: List[int]
    n_pilot_symbols: int
    n_data_symbols: int      # OFDM symbols carrying data
    num_data_re: int         # data resource elements per stream
    n_coded: int             # LDPC coded bits per stream  (n)
    k_info: int              # LDPC info bits per stream    (k)
    K: int                   # total info bits across streams (== batch.bits dim)


def infer_dims(config) -> Dims:
    """Derive all code/grid dimensions from ``config`` (no Sionna dependency)."""
    phy = config.phy
    n_sc = int(phy.n_subcarriers)
    n_sym = int(phy.n_symbols)
    n_rx = int(phy.n_rx)
    n_tx = int(phy.n_tx)
    n_bps = int(phy.n_bits_per_symbol)
    rate = float(phy.ldpc_coderate)

    # OmegaConf supports .get with a default.
    pilot_idx = list(phy.get("pilot_ofdm_symbol_indices", [2, 11]))
    n_pilot = len(pilot_idx)
    n_data_sym = n_sym - n_pilot
    if n_data_sym <= 0:
        raise ValueError(
            f"n_pilot_symbols ({n_pilot}) >= n_symbols ({n_sym}); no data symbols."
        )

    num_data_re = n_sc * n_data_sym          # per stream
    n_coded = num_data_re * n_bps            # per stream  (LDPC n)
    k_info = int(n_coded * rate)             # per stream  (LDPC k)
    K = k_info * n_tx                        # total info bits

    return Dims(
        n_subcarriers=n_sc,
        n_symbols=n_sym,
        n_rx=n_rx,
        n_tx=n_tx,
        n_bits_per_symbol=n_bps,
        coderate=rate,
        pilot_symbol_indices=pilot_idx,
        n_pilot_symbols=n_pilot,
        n_data_symbols=n_data_sym,
        num_data_re=num_data_re,
        n_coded=n_coded,
        k_info=k_info,
        K=K,
    )


def data_symbol_indices(config) -> List[int]:
    """Indices of OFDM symbols that carry data (i.e. are *not* pilot symbols)."""
    d = infer_dims(config)
    pilots = set(d.pilot_symbol_indices)
    return [s for s in range(d.n_symbols) if s not in pilots]
