"""Receiver-verifiable transport-block CRC on top of :class:`SionnaChannel`.

The transmitter draws a payload of ``A`` bits, appends an ``L``-bit CRC and
LDPC-encodes the resulting ``K = A + L`` bits. The receiver recomputes the CRC
on its own decoded bits, so any adaptation gate built on :meth:`crc_check`
uses receiver-side information only (never the transmitted bits).

CRC selection follows 3GPP TS 38.212, Sec. 7.2.1: CRC16 when the payload
``A <= 3824``, CRC24A otherwise. For the configurations used here
(K = 1824 for 16-QAM, K = 2736 for 64-QAM) this yields CRC16.
"""

from __future__ import annotations

from typing import Optional

import torch

from sionna.phy.fec.crc import CRCEncoder, CRCDecoder

from src.channel.sionna_channel import SionnaChannel


def nr_crc_degree(k_total: int) -> str:
    """CRC polynomial for a transport block of ``k_total`` = A + L bits."""
    return "CRC16" if (k_total - 16) <= 3824 else "CRC24A"


class _CRCSource:
    """Binary source that emits CRC-valid information words of length K."""

    def __init__(self, source, encoder: CRCEncoder, k_payload: int):
        self.source = source
        self.encoder = encoder
        self.k_payload = k_payload

    def __call__(self, shape):
        shape = list(shape)
        shape[-1] = self.k_payload
        return self.encoder(self.source(shape))


class CRCSionnaChannel(SionnaChannel):
    """:class:`SionnaChannel` whose information words carry a real CRC."""

    def __init__(self, config, device: Optional[str] = None,
                 crc_degree: Optional[str] = None, **kwargs):
        super().__init__(config, device=device, **kwargs)
        self.crc_degree = crc_degree or nr_crc_degree(self.k)
        probe = CRCEncoder(self.crc_degree, device=self.device)
        self.crc_length = int(probe.crc_length)
        self.k_payload = self.k - self.crc_length
        self.crc_encoder = CRCEncoder(self.crc_degree, k=self.k_payload,
                                      device=self.device)
        self.crc_decoder = CRCDecoder(self.crc_encoder, device=self.device)
        self.binary_source = _CRCSource(self.binary_source, self.crc_encoder,
                                        self.k_payload)

    @torch.no_grad()
    def crc_check(self, info_bits_hat: torch.Tensor) -> torch.Tensor:
        """CRC verdict per block from the receiver's hard decisions.

        Args:
            info_bits_hat: ``(B, K)`` hard decisions in {0, 1}.
        Returns:
            ``(B,)`` bool, True where the CRC is satisfied.
        """
        B = info_bits_hat.shape[0]
        x = info_bits_hat.to(self.device, torch.float32).reshape(B, 1, self.n_tx, self.k)
        _, valid = self.crc_decoder(x)
        return valid.reshape(B, -1).all(dim=1)
