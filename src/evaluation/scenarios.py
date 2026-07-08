"""
src/evaluation/scenarios.py

Phase 8 — channel-scenario helpers (S1-S4).

The scenario *definitions* live in ``config.scenarios`` (config.yaml). This
module turns a scenario key into a built :class:`SionnaChannel`, mapping the
config field names to the channel constructor arguments:

    channel_model   -> cdl_model
    delay_spread_ns -> delay_spread_ns
    ue_speed_kmh    -> ue_speed_kmh

  * **S1** ``s1_matched``     — CDL-C, 100 ns, 3 km/h  (training distribution).
  * **S2** ``s2_delay_shift`` — CDL-A, 1000 ns, 3 km/h (delay-spread drift).
  * **S3** ``s3_speed_shift`` — CDL-C, 100 ns, 90 km/h (Doppler drift).
  * **S4** ``s4_ray_traced``  — Sionna RT scene (e.g. Munich). Not CDL-based;
    requires the ray-tracing pipeline and is treated as optional ("if
    available", CLAUDE.md Phase 8) — :func:`build_scenario_channel` raises a
    clear, catchable error so the evaluation loop can skip it.
"""

from __future__ import annotations

from typing import List, Optional

from src.channel.sionna_channel import SionnaChannel

# Scenario keys that are CDL-based and therefore directly supported here.
CDL_SCENARIO_KEYS: List[str] = ["s1_matched", "s2_delay_shift", "s3_speed_shift"]

# Human-readable labels for tables / plots.
SCENARIO_LABELS = {
    "s1_matched": "S1",
    "s2_delay_shift": "S2",
    "s3_speed_shift": "S3",
    "s4_ray_traced": "S4",
}


def is_cdl_scenario(scenario_cfg) -> bool:
    """True if the scenario is CDL-based (has a ``channel_model`` field)."""
    try:
        return "channel_model" in scenario_cfg
    except TypeError:
        return hasattr(scenario_cfg, "channel_model")


def build_scenario_channel(
    config, scenario_key: str, device: Optional[str] = None
) -> SionnaChannel:
    """Build the :class:`SionnaChannel` for a named scenario.

    Raises:
        KeyError:            unknown scenario key.
        NotImplementedError: non-CDL scenario (e.g. ray-traced S4) — caller
                             should catch this and skip the scenario.
    """
    if scenario_key not in config.scenarios:
        raise KeyError(f"Unknown scenario '{scenario_key}'.")
    sc = config.scenarios[scenario_key]

    if not is_cdl_scenario(sc):
        raise NotImplementedError(
            f"Scenario '{scenario_key}' is not CDL-based (e.g. ray-traced). "
            "The ray-tracing pipeline is not wired into this evaluation; skip it."
        )

    return SionnaChannel(
        config,
        device=device,
        cdl_model=sc.channel_model,
        delay_spread_ns=sc.get("delay_spread_ns", None),
        ue_speed_kmh=sc.get("ue_speed_kmh", None),
    )


def scenario_label(scenario_key: str) -> str:
    return SCENARIO_LABELS.get(scenario_key, scenario_key)
