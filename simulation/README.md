# Simulation

Simulated three-UAV deployment used in the final-review deck. Every position estimate comes from the
project's own solver (`backend/trilateration.py`), fed through the same 5-frame RSSI average as
`backend/utils.py`, with node positions read from `config/node_positions.json`.

```bash
pip install numpy matplotlib imageio-ffmpeg
cd simulation
python montecarlo.py accuracy_map.png mc_stats.json      # accuracy study (~30 s)
python fig_accuracy.py mc_stats_med.npy accuracy_map.png  # re-plot the map only
python render_sim.py video simulation.mp4                 # 42 s, 1080p30 H.264
python render_sim.py frames . 8 20.9 38                   # stills at given seconds
```

Assumptions: 4 dB log-normal shadowing per frame, backend defaults (−40 dBm at 1 m, n = 2.0),
time accelerated (phones probe ~0.9×/s here). `SIM_SEED` picks the run; the deck uses 23.
