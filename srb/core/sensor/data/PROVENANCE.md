# `livox_mid360_scan_pattern.npz`

Source: `Livox-SDK/livox_laser_simulation`, `scan_mode/mid360.csv`
(https://github.com/Livox-SDK/livox_laser_simulation), MIT License,
Copyright (c) 2021 livox.

Livox's official Gazebo-plugin scan pattern for the Mid-360 lidar: one full
4.0s non-repetitive scan cycle at the sensor's 200,000 pts/s point rate
(800,000 samples). Converted 2026-09-14 from the raw CSV
(`Time/s,Azimuth/deg,Zenith/deg`) via:

```python
elevation_deg = 90.0 - zenith_deg  # CSV's zenith is angle-from-vertical
azimuth_deg = ((azimuth_deg + 180.0) % 360.0) - 180.0  # wrap to [-180, 180)
```

Arrays (float32, shape `(800000,)`, in original CSV sample order):
`azimuth_deg`, `elevation_deg`.

The real Mid-360/its Gazebo plugin replays this as a sliding window of
24,000 consecutive samples per tick, advancing and wrapping every tick
(`srb/core/sensor/rtx_lidar.py` reproduces this scheme, not a single static
pattern) -- see `docs/rtx_lidar_integration_plan.md` in the research repo
for the full derivation.
