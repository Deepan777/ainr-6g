"""
src/channel/sionna_channel.py

Phase 1 — Sionna 2.x channel wrapper.

Generates batches of ``(bits, channel_response, received_grid)`` triples using a
5G NR OFDM uplink chain:

    bits -> LDPC encode -> QAM map -> resource grid -> CDL channel -> + AWGN

All tensors are returned as PyTorch tensors on the configured device.

Sionna 2.x notes (API changed from the TensorFlow-based v1):
  * Everything lives under ``sionna.phy.*`` and is PyTorch-native.
  * Devices must be ``'cpu'`` or ``'cuda:0'`` (the bare string ``'cuda'`` is
    rejected by ``sionna.phy.config``), so we normalise it here.
  * Channel models (CDL) are paired with ``GenerateOFDMChannel`` /
    ``ApplyOFDMChannel`` instead of the old ``OFDMChannel`` callable layer.

Modelling choices (UE -> gNB uplink):
  * One transmitter (UE) with ``n_tx`` antennas (default 1).
  * One receiver (gNB) with ``n_rx`` antennas (default 4), modelled in Sionna as
    a single receiver (``num_rx = 1``) carrying ``num_rx_ant = n_rx`` antennas.

Output shapes (see CLAUDE.md Phase 1):
  * ``bits``             : ``(B, K)``                       — info bits {0, 1}
  * ``channel_response`` : ``(B, N_rx, N_tx, N_sc, N_sym)`` — complex freq. response
  * ``received_grid``    : ``(B, N_sc, N_sym, N_rx)``       — complex received grid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from sionna.phy.channel.tr38901 import CDL, PanelArray
from sionna.phy.channel import GenerateOFDMChannel, ApplyOFDMChannel
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder


# CDL "CDL-C" -> Sionna expects just the letter "C".
def _cdl_letter(model_name: str) -> str:
    """Map a config CDL name (e.g. 'CDL-C' or 'C') to Sionna's single letter."""
    name = str(model_name).upper().strip()
    if name.startswith("CDL-"):
        name = name.split("-", 1)[1]
    if name not in {"A", "B", "C", "D", "E"}:
        raise ValueError(f"Unsupported CDL model '{model_name}'. Expected A-E.")
    return name


def _normalise_device(device: Optional[str]) -> str:
    """Sionna only accepts 'cpu' or 'cuda:0' — turn 'cuda' into 'cuda:0'."""
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


@dataclass
class ChannelBatch:
    """A batch of generated channel data. Every tensor lives on ``device``."""

    bits: torch.Tensor              # (B, K)                     float {0, 1}
    channel_response: torch.Tensor  # (B, N_rx, N_tx, N_sc, N_sym) complex
    received_grid: torch.Tensor     # (B, N_sc, N_sym, N_rx)     complex
    no: torch.Tensor                # (B,)                       float noise variance
    snr_db: torch.Tensor            # (B,)                       float SNR (dB)
    # Native-Sionna-layout tensors, kept so the generative model (Phase 3) can
    # reuse the exact same forward operators without reshuffling axes.
    h_freq: torch.Tensor            # (B, num_rx, num_rx_ant, num_tx, num_tx_ant, N_sym, N_sc)
    tx_grid: torch.Tensor           # (B, num_tx, num_streams, N_sym, N_sc) complex


class SionnaChannel:
    """Generates 5G NR OFDM uplink batches with a CDL channel.

    Args:
        config:           OmegaConf config (uses ``phy`` and ``channel`` groups).
        device:           'cpu' or 'cuda' / 'cuda:0'.
        cdl_model:        Override CDL model letter/name (A-E). Defaults to config.
        delay_spread_ns:  Override delay spread (ns). Defaults to config.
        ue_speed_kmh:     Fixed UE speed (km/h). If given, overrides the config
                          training speed *range* with a single value (used for the
                          evaluation scenarios S1-S4). If ``None``, the config
                          ``[min, max]`` range is used and a speed is drawn per batch.
    """

    def __init__(
        self,
        config,
        device: Optional[str] = None,
        cdl_model: Optional[str] = None,
        delay_spread_ns: Optional[float] = None,
        ue_speed_kmh: Optional[float] = None,
    ):
        self.config = config
        self.device = _normalise_device(device)

        phy = config.phy
        chan = config.channel

        # ---- Physical-layer dimensions ---------------------------------
        self.n_subcarriers = int(phy.n_subcarriers)
        self.n_symbols = int(phy.n_symbols)
        self.n_rx = int(phy.n_rx)                       # gNB antennas
        self.n_tx = int(phy.n_tx)                       # UE antennas
        self.n_bits_per_symbol = int(phy.n_bits_per_symbol)
        self.coderate = float(phy.ldpc_coderate)
        self.carrier_frequency = float(chan.carrier_frequency)

        # ---- Channel scenario parameters -------------------------------
        self.cdl_model = _cdl_letter(cdl_model if cdl_model is not None else chan.model)
        self.delay_spread = (
            float(delay_spread_ns) if delay_spread_ns is not None
            else float(chan.delay_spread_ns)
        ) * 1e-9
        if ue_speed_kmh is not None:
            self.min_speed = self.max_speed = float(ue_speed_kmh) / 3.6  # km/h -> m/s
        else:
            self.min_speed = float(chan.ue_speed_kmh_min) / 3.6
            self.max_speed = float(chan.ue_speed_kmh_max) / 3.6

        # Training SNR range (used when generate_batch is called without snr_db).
        self.snr_db_min = float(chan.snr_db_min)
        self.snr_db_max = float(chan.snr_db_max)

        # ---- Build the Sionna processing blocks ------------------------
        self._build_blocks()

    # ------------------------------------------------------------------ #
    #  Construction helpers
    # ------------------------------------------------------------------ #
    def _build_antenna_arrays(self):
        """UE (single-pol) and gNB (dual-pol panel) antenna arrays."""
        dev = self.device
        fc = self.carrier_frequency

        # UE transmit array: n_tx single-polarised omni elements.
        self.ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=self.n_tx,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=fc,
            device=dev,
        )

        # gNB receive array: dual-polarised panel giving exactly n_rx elements.
        # Dual polarisation contributes a factor of 2, so we need n_rx/2 columns.
        if self.n_rx % 2 == 0:
            num_cols = self.n_rx // 2
            polarization = "dual"
            polarization_type = "VH"
        else:
            num_cols = self.n_rx
            polarization = "single"
            polarization_type = "V"
        self.bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=num_cols,
            polarization=polarization,
            polarization_type=polarization_type,
            antenna_pattern="38.901",
            carrier_frequency=fc,
            device=dev,
        )
        if self.bs_array.num_ant != self.n_rx:
            raise ValueError(
                f"gNB array produced {self.bs_array.num_ant} antennas, "
                f"expected n_rx={self.n_rx}."
            )

    def _build_blocks(self):
        dev = self.device
        self._build_antenna_arrays()

        # ---- OFDM resource grid (5G NR slot) ---------------------------
        # Pilots on full OFDM symbols (DMRS-like) using the Kronecker pattern.
        pilot_indices = list(
            self.config.phy.get("pilot_ofdm_symbol_indices", [2, 11])
        )
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.n_symbols,
            fft_size=self.n_subcarriers,
            subcarrier_spacing=30e3,
            num_tx=1,
            num_streams_per_tx=self.n_tx,
            cyclic_prefix_length=6,
            pilot_pattern=self.config.phy.pilot_pattern,
            pilot_ofdm_symbol_indices=pilot_indices,
            device=dev,
        )
        self.num_data_symbols = int(self.resource_grid.num_data_symbols)

        # Deterministic pilots. Sionna randomises pilot values per ResourceGrid
        # instance, but in a real link the pilots are a *known* reference shared
        # by transmitter and receiver. Fixing them (same fixed seed for every
        # instance) makes all channel instances agree, so the classical LMMSE
        # estimator and the AINR generative model decode data generated by *any*
        # instance — not only the one they were constructed from.
        self._set_deterministic_pilots(dev)

        # ---- LDPC code sizing ------------------------------------------
        # Coded bits fill every data resource element; info bits = rate * n.
        self.n = self.num_data_symbols * self.n_bits_per_symbol
        self.k = int(self.n * self.coderate)

        # Cross-check against the config-only derivation used by modules that
        # never see the resource grid (e.g. VariationalPosterior). If these ever
        # diverge, the bit-LLR head would be mis-sized and decoding would break.
        from src.utils import infer_dims
        _d = infer_dims(self.config)
        if (_d.num_data_re, _d.n_coded, _d.k_info) != (
            self.num_data_symbols, self.n, self.k
        ):
            raise ValueError(
                "infer_dims() disagrees with the Sionna resource grid: "
                f"config-derived (data_re={_d.num_data_re}, n={_d.n_coded}, "
                f"k={_d.k_info}) vs grid (data_re={self.num_data_symbols}, "
                f"n={self.n}, k={self.k}). Check phy.pilot_ofdm_symbol_indices."
            )

        # ---- Forward-chain blocks --------------------------------------
        self.binary_source = BinarySource(device=dev)
        self.encoder = LDPC5GEncoder(self.k, self.n, device=dev)
        self.decoder = LDPC5GDecoder(
            self.encoder, hard_out=True, return_infobits=True,
            num_iter=20, device=dev,
        )
        self.mapper = Mapper("qam", self.n_bits_per_symbol, device=dev)
        self.demapper = Demapper("app", "qam", self.n_bits_per_symbol, device=dev)
        self.rg_mapper = ResourceGridMapper(self.resource_grid, device=dev)

        # ---- Channel ----------------------------------------------------
        self.cdl = CDL(
            model=self.cdl_model,
            delay_spread=self.delay_spread,
            carrier_frequency=self.carrier_frequency,
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction="uplink",
            min_speed=self.min_speed,
            max_speed=self.max_speed,
            device=dev,
        )
        self.generate_channel = GenerateOFDMChannel(
            self.cdl, self.resource_grid, normalize_channel=True, device=dev,
        )
        self.apply_channel = ApplyOFDMChannel(device=dev)

        # ---- Pilot reference grid (for the neural receivers' input) -----
        # Mapping zero data symbols leaves only the (known, deterministic) pilots
        # on the grid. Stored in CLAUDE.md layout (N_sc, N_sym) so receivers can
        # form a least-squares channel estimate H_ls = Y * conj(pilot) at the
        # pilot REs — the standard "pilot reference" input of neural receivers.
        _zeros = torch.zeros(
            1, 1, self.n_tx, self.num_data_symbols, dtype=torch.complex64, device=dev
        )
        _pg = self.rg_mapper(_zeros)                 # (1,1,n_tx,N_sym,N_sc)
        self.pilot_grid = _pg[0, 0, 0].transpose(0, 1).contiguous()  # (N_sc, N_sym)

    def _set_deterministic_pilots(self, device) -> None:
        """Overwrite the (randomised) pilot symbols with a fixed QPSK sequence.

        Uses a constant seed so every :class:`SionnaChannel` instance produces
        identical, unit-modulus pilots — keeping the pilot reference consistent
        across instances (see note in :meth:`_build_blocks`).
        """
        import math

        pilots = self.resource_grid.pilot_pattern.pilots
        gen = torch.Generator().manual_seed(20240529)
        idx = torch.randint(0, 4, tuple(pilots.shape), generator=gen).to(torch.float32)
        angle = math.pi / 4.0 + idx * (math.pi / 2.0)        # QPSK constellation
        qpsk = torch.polar(torch.ones_like(angle), angle)    # unit-modulus complex
        self.resource_grid.pilot_pattern.pilots = qpsk.to(device=device, dtype=pilots.dtype)

    # ------------------------------------------------------------------ #
    #  Public properties
    # ------------------------------------------------------------------ #
    @property
    def num_info_bits(self) -> int:
        """K — number of LDPC information bits per codeword (per stream)."""
        return self.k * self.n_tx

    @property
    def num_coded_bits(self) -> int:
        """n — number of LDPC coded bits per codeword (per stream)."""
        return self.n

    # ------------------------------------------------------------------ #
    #  SNR / noise utilities
    # ------------------------------------------------------------------ #
    def snr_db_to_no(self, snr_db: torch.Tensor) -> torch.Tensor:
        """Convert an (Es/N0) SNR in dB to a linear noise variance ``no``.

        The channel is energy-normalised and QAM symbols have unit average
        power, so the per-resource-element SNR is ``1 / no``.
        """
        return torch.pow(10.0, -snr_db / 10.0)

    def _sample_snr_db(self, batch_size: int, snr_db) -> torch.Tensor:
        """Return a (B,) SNR tensor, sampling from the training range if needed."""
        if snr_db is None:
            u = torch.rand(batch_size, device=self.device)
            return self.snr_db_min + u * (self.snr_db_max - self.snr_db_min)
        if torch.is_tensor(snr_db):
            snr = snr_db.to(self.device).float()
            if snr.ndim == 0:
                snr = snr.expand(batch_size).clone()
            return snr
        return torch.full((batch_size,), float(snr_db), device=self.device)

    # ------------------------------------------------------------------ #
    #  Forward-model helpers (reused by the generative model in Phase 3)
    # ------------------------------------------------------------------ #
    def transmit(self, info_bits: torch.Tensor) -> torch.Tensor:
        """Map info bits to the transmitted OFDM resource grid.

        Args:
            info_bits: (B, K) float {0, 1}.

        Returns:
            tx_grid: (B, num_tx, num_streams, N_sym, N_sc) complex.
        """
        B = info_bits.shape[0]
        # Reshape (B, K) -> (B, num_tx=1, num_streams=n_tx, k)
        bits = info_bits.reshape(B, 1, self.n_tx, self.k)
        codeword = self.encoder(bits)                 # (B, 1, n_tx, n)
        symbols = self.mapper(codeword)               # (B, 1, n_tx, n_data)
        tx_grid = self.rg_mapper(symbols)             # (B, 1, n_tx, N_sym, N_sc)
        return tx_grid

    def apply_channel_freq(
        self,
        tx_grid: torch.Tensor,
        h_freq: torch.Tensor,
        no: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Apply a frequency-domain channel + AWGN to a transmitted grid.

        Returns the received grid in *Sionna* layout
        ``(B, num_rx, num_rx_ant, N_sym, N_sc)``.
        """
        return self.apply_channel(tx_grid, h_freq, no)

    # ------------------------------------------------------------------ #
    #  Layout conversions  (Sionna native  <->  CLAUDE.md convention)
    # ------------------------------------------------------------------ #
    def _channel_to_claude_layout(self, h_freq: torch.Tensor) -> torch.Tensor:
        """h_freq (B, 1, N_rx, 1, 1, N_sym, N_sc) -> (B, N_rx, N_tx, N_sc, N_sym)."""
        # Collapse the (num_rx=1) and (num_tx_ant=1) singleton axes.
        # h_freq dims: [B, num_rx, num_rx_ant, num_tx, num_tx_ant, N_sym, N_sc]
        h = h_freq[:, 0, :, :, 0, :, :]      # (B, N_rx, N_tx, N_sym, N_sc)
        h = h.permute(0, 1, 2, 4, 3)         # (B, N_rx, N_tx, N_sc, N_sym)
        return h.contiguous()

    def _received_to_claude_layout(self, y: torch.Tensor) -> torch.Tensor:
        """y (B, num_rx=1, N_rx, N_sym, N_sc) -> (B, N_sc, N_sym, N_rx)."""
        y = y[:, 0]                          # (B, N_rx, N_sym, N_sc)
        y = y.permute(0, 3, 2, 1)            # (B, N_sc, N_sym, N_rx)
        return y.contiguous()

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #
    def generate_batch(
        self,
        batch_size: int,
        snr_db: Optional[Union[float, torch.Tensor]] = None,
    ) -> ChannelBatch:
        """Generate one batch of (bits, channel, received grid).

        Args:
            batch_size: number of independent slots to generate.
            snr_db:     fixed SNR (dB) for the whole batch, or ``None`` to draw a
                        per-example SNR uniformly from the configured range.

        Returns:
            ChannelBatch with tensors in CLAUDE.md layout.
        """
        B = int(batch_size)

        # 1) Information bits.
        info_bits = self.binary_source([B, 1, self.n_tx, self.k])  # (B,1,n_tx,k)

        # 2) Transmit chain -> resource grid.
        codeword = self.encoder(info_bits)            # (B,1,n_tx,n)
        symbols = self.mapper(codeword)               # (B,1,n_tx,n_data)
        tx_grid = self.rg_mapper(symbols)             # (B,1,n_tx,N_sym,N_sc)

        # 3) Channel realisation (fresh per call).
        h_freq = self.generate_channel(B)             # (B,1,N_rx,1,1,N_sym,N_sc)

        # 4) Noise + channel application.
        snr = self._sample_snr_db(B, snr_db)          # (B,)
        no = self.snr_db_to_no(snr)                   # (B,)
        no_b = no.reshape(B, 1, 1, 1, 1)              # broadcastable to received grid
        y = self.apply_channel(tx_grid, h_freq, no_b)  # (B,1,N_rx,N_sym,N_sc)

        # 5) Re-layout to CLAUDE.md convention.
        bits_out = info_bits.reshape(B, self.num_info_bits).float()
        channel_response = self._channel_to_claude_layout(h_freq)
        received_grid = self._received_to_claude_layout(y)

        return ChannelBatch(
            bits=bits_out,
            channel_response=channel_response,
            received_grid=received_grid,
            no=no,
            snr_db=snr,
            h_freq=h_freq,
            tx_grid=tx_grid,
        )
