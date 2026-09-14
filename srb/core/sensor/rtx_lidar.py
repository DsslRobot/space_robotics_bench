"""Custom RTX Lidar sensor support (Livox Mid-360).

Not an IsaacLab sensor. IsaacLab's team evaluated wrapping Isaac Sim's RTX
Lidar (`isaacsim.sensors.rtx`) and dropped it because it doesn't batch
across parallel envs the way the Warp-based `RayCaster` does
(isaac-sim/IsaacLab discussion #1201) -- irrelevant for this project (no
`num_envs` parallelism requirement, see docs/rtx_lidar_integration_plan.md
in the research repo). This module authors a custom Mid-360 lidar directly
against `isaacsim.sensors.rtx` / `omni.replicator.core`, bypassing
IsaacLab's sensor abstraction (and, per that same investigation, the
`IsaacSensorCreateRtxLidar` command's vendor-asset whitelist -- which does
not apply to this authoring path at all; see the plan doc's derivation
from the actual USD schema).

Status (2026-09-14): the sensor authoring path below (`build_mid360_lidar`)
is verified live -- an `OmniLidar` prim with custom
`omni:sensor:Core:emitterState:*` attributes is created correctly, schema
applied, attribute values round-trip exactly as set. Pulling *live point
cloud data* through it is currently blocked by a confirmed upstream Isaac
Sim bug, not anything in this module: RTX Lidar annotators return empty
data in headless mode on this pinned build (5.1.0-rc.19). Reproduced with
both this custom sensor AND an official whitelisted reference sensor
(HESAI_XT32_SD10) -- same empty-data symptom either way, so it is not
specific to a custom sensor. Matches isaac-sim/IsaacSim GitHub issue #110
("Cannot retrieve RTX lidar data using headless Python script") and an
NVIDIA forum thread reporting the same symptom
(forums.developer.nvidia.com/t/unable-to-reliably-produce-lidar-data-in-a-headless-simulation/337835),
where an NVIDIA moderator stated the issue was fixed in Isaac Sim 5.1.0 --
but our pinned build is an RC (release candidate) cut before that fix,
not the GA release. Confirmed *not* fixed by the two documented headless
workarounds either (`rt_subframes>=2`,
`--/exts/isaacsim.core.throttling/enable_async=false`, both from Isaac
Sim's own Known Issues page) -- the render pipeline's
`/Render/PostProcess/SDGPipeline/PostProcessDispatcher` node never
advances past reference time 0/0 on this build, regardless of how the
render step is driven (bare `simulation_app.update()`,
`SimulationContext.step()`, the async `next_render_simulation_async`, or
the purpose-built synchronous `rep.orchestrator.step()` -- all six
combinations tried, all show the identical symptom).

This is therefore infrastructure that is authored correctly and ready, but
not yet wired into `srb/core/env/mobile/env.py` as the active `lidar_robot`
sensor (RayCaster stays the active implementation) until this upstream
bug is confirmed resolved on whatever Isaac Sim build this project moves
to next. See docs/rtx_lidar_integration_plan.md for the full plan and
re-verification steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd

_DATA_DIR = Path(__file__).parent / "data"
_MID360_SCAN_PATTERN_PATH = _DATA_DIR / "livox_mid360_scan_pattern.npz"

## Livox Mid-360 datasheet (https://www.livoxtech.com/mid-360/specs,
## verified 2026-09-14): FOV 360 deg horizontal, -7..+52 deg vertical;
## 40 m detection range @ 10% reflectivity (70 m @ 80%, not modeled here --
## a single conservative farRangeM is used); 0.1 m blind zone; 905 nm
## Class-1 eye-safe wavelength; 200,000 pts/s point rate.
MID360_NEAR_RANGE_M = 0.1
MID360_FAR_RANGE_M = 40.0
MID360_WAVELENGTH_NM = 905.0
MID360_POINT_RATE_HZ = 200_000.0
MID360_RANGE_ACCURACY_M = 0.02

## The real sensor (and Livox's own Gazebo plugin, `livox_laser_simulation`,
## `scan_mode/mid360.csv` -- see srb/core/sensor/data/PROVENANCE.md) replays
## a full 800,000-sample, 4.0s scan cycle as a sliding window of this many
## consecutive samples per tick, advancing and wrapping every tick -- not a
## single static pattern.
MID360_WINDOW_SIZE = 24_000


class Mid360ScanPattern:
    """Loads the real Mid-360 scan pattern and slices it into the same
    sequential, wrapping windows the actual hardware/simulation plugin
    uses, so consecutive ticks see genuinely different point directions
    instead of one repeating static pattern."""

    def __init__(self, window_size: int = MID360_WINDOW_SIZE):
        data = np.load(_MID360_SCAN_PATTERN_PATH)
        self.azimuth_deg: np.ndarray = data["azimuth_deg"]
        self.elevation_deg: np.ndarray = data["elevation_deg"]
        assert self.azimuth_deg.shape == self.elevation_deg.shape
        self.n_samples = self.azimuth_deg.shape[0]
        self.window_size = window_size
        self.n_windows = -(-self.n_samples // window_size)  # ceil div

    def window(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (azimuth_deg, elevation_deg, fire_time_ns) for window
        `index`, wrapping via modulo `n_windows` -- matching the real
        plugin's `currStartIndex += samples; % maxPointSize` scheme."""
        w = index % self.n_windows
        start = w * self.window_size
        end = min(start + self.window_size, self.n_samples)
        az = self.azimuth_deg[start:end]
        el = self.elevation_deg[start:end]
        # Real point spacing at the sensor's 200,000 pts/s rate.
        fire_time_ns = (np.arange(az.shape[0]) * (1e9 / MID360_POINT_RATE_HZ)).astype(
            np.uint32
        )
        return az, el, fire_time_ns


def mid360_emitter_kwargs(
    az: np.ndarray, el: np.ndarray, fire_time_ns: np.ndarray, instance: str = "s001"
) -> dict:
    """Build the `omni:sensor:Core:*` kwargs for
    `rep.functional.create.omni_lidar` / `rep.functional.create_batch.omni_lidar`
    for one emitter-state window. Verified live 2026-09-14: the prim is
    created with `OmniSensorGenericLidarCoreAPI` applied and these array
    values round-trip exactly."""
    n = az.shape[0]
    prefix = f"omni:sensor:Core:emitterState:{instance}"
    return {
        "omni:sensor:Core:scanType": "SOLID_STATE",
        "omni:sensor:Core:nearRangeM": MID360_NEAR_RANGE_M,
        "omni:sensor:Core:farRangeM": MID360_FAR_RANGE_M,
        "omni:sensor:Core:waveLengthNm": MID360_WAVELENGTH_NM,
        "omni:sensor:Core:rangeAccuracyM": MID360_RANGE_ACCURACY_M,
        "omni:sensor:Core:reportRateBaseHz": int(MID360_POINT_RATE_HZ),
        "omni:sensor:Core:scanRateBaseHz": 10,
        "omni:sensor:Core:numberOfEmitters": n,
        "omni:sensor:Core:numberOfChannels": n,
        "omni:sensor:Core:numLines": 1,
        "omni:sensor:Core:numRaysPerLine": [n],
        f"{prefix}:azimuthDeg": az.tolist(),
        f"{prefix}:elevationDeg": el.tolist(),
        f"{prefix}:fireTimeNs": fire_time_ns.tolist(),
        f"{prefix}:channelId": list(range(n)),
    }


def build_mid360_lidar(
    prim_path: str,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    window_index: int = 0,
) -> "Usd.Prim":
    """Author a custom Mid-360 `OmniLidar` prim (one scan-pattern window).

    Bypasses IsaacLab's sensor abstraction and `IsaacSensorCreateRtxLidar`'s
    vendor-asset whitelist entirely -- verified live 2026-09-14 to
    correctly create the prim, apply `OmniSensorGenericLidarCoreAPI`, and
    round-trip the emitter arrays. Advancing `window_index` and re-setting
    the emitter attributes (`prim.GetAttribute(...).Set(...)`, values only,
    not array length -- see module docstring) each tick reproduces the
    real sliding-window scan; see `Mid360ScanPattern.window()`.
    """
    import omni.replicator.core as rep

    pattern = Mid360ScanPattern()
    az, el, fire_time_ns = pattern.window(window_index)
    kwargs = mid360_emitter_kwargs(az, el, fire_time_ns)
    prim = rep.functional.create.omni_lidar(
        position=position, rotation=rotation_deg, name=Path(prim_path).name, **kwargs
    )
    return prim
