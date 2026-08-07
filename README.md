# toolbox

[![toolbox import check](https://github.com/drozzoff/toolbox/actions/workflows/import-check.yml/badge.svg)](https://github.com/drozzoff/toolbox/actions/workflows/import-check.yml)

Research utilities built around [Xsuite](https://xsuite.readthedocs.io/) for SIS18 beam dynamics, slow extraction, tracking, diagnostics, and visualization.

The package collects tools used in simulation and analysis workflows, including:

- multi-GPU particle tracking;
- tune-spectrum analysis;
- SIS18 BPSK excitation and rigidity ramps;
- slow-extraction and mixed-beam dashboards;
- separatrix and phase-space analysis;
- multi-species accelerating beamlines;
- lattice plotting and aperture helpers.

> This is research software under active development. The API may change between versions.

## Requirements

- Python 3.10 or newer
- Xsuite/Xtrack
- NumPy, SciPy, and pandas
- Matplotlib, Plotly, Seaborn, and Dash
- HDF5 support through `h5py`

Multi-GPU tracking additionally requires an OpenCL-capable environment supported by `xobjects.ContextPyopencl`.

## Installation

Clone and install the repository:

```bash
git clone https://github.com/drozzoff/toolbox.git
cd toolbox
python -m pip install .
```

For development, use an editable installation:

```bash
python -m pip install -e .
```

## Main functionality

| Area | Public interface |
|---|---|
| Multi-GPU tracking | `toolbox.track_multigpu()` |
| Frequency-chirp exciter | `toolbox.exc_freq_chirp()` |
| Multi-species lines | `toolbox.create_multispecies_lines()` |
| Tune analysis | `toolbox.SIS18.parse_tunes()` |
| BPSK excitation | `toolbox.SIS18.plain_bpsk()`, `toolbox.SIS18.modulated_bpsk()` |
| SIS18 rigidity ramps | `toolbox.SIS18.get_rigidity_ramp()` |
| Extraction dashboard | `toolbox.dashboard.Dashboard` |
| Phase-space snapshots | `toolbox.PhaseSpaceSnapshots` |
| Separatrix analysis | `toolbox.get_stable_limit()`, `toolbox.get_separatrix_vertices()` |
| Lattice plotting | `toolbox.PlotContext` |
| Aperture correction | `toolbox.realign_mad_apertures()` |

## Tune-spectrum analysis

`parse_tunes()` identifies a single spectral peak at each timestamp, fits it with a Gaussian, and filters fits using their weighted R2 score.

The input should be a pandas `DataFrame` whose rows represent timestamps and whose columns represent frequency or tune bins.

```python
import pandas as pd
import toolbox as tb

spectrogram = pd.read_pickle("horizontal_spectrum.pkl")

result = tb.SIS18.parse_tunes(
    data = spectrogram,
    noise_percentile = 0.99,
    search_region_width = 5,
    noise_width = 3,
    peak_min_length = 1,
    post_dilation = 3,
    weights_func = tb.SIS18.rational_weight,
    r2_score_cut = 0.5,
    verbose = 1,
)

print(result)
```

The result contains the fitted peak center, its uncertainty, and the fit score.

## SIS18 BPSK excitation

```python
import numpy as np
from toolbox.SIS18 import plain_bpsk

timestamps = np.arange(0, 1.0, 1e-6)

signal = plain_bpsk(
    frev = 1.0e6,
    Qx = 0.33,
    Qx_bandwidth = 0.01,
    timestamps = timestamps,
    filename = "bpsk_signal.npy",
    verbose = 1,
)
```

`modulated_bpsk()` additionally applies the SIS18-style amplitude program and returns the maximum peak-to-peak voltage together with the normalized signal.

## Dashboard

The dashboard can display live newline-delimited JSON received over TCP or load data through a selected SIS18 profile.

```python
from toolbox.dashboard import Dashboard, SIS18extraction

dashboard = Dashboard(
    profile = SIS18extraction(start_count_at_turn = 0),
    host = "127.0.0.1",
    port = 35235,
    data_to_monitor = [
        "intensity",
        "spill",
        "spill:accumulated",
        "ES_septum_losses",
        "ES_entrance_phase_space",
    ],
)

dashboard.start_listener()
dashboard.run_dash_server()
```

The TCP listener uses port `35235` in this example. The Dash application is served separately on Dash's default address, normally:

```text
http://127.0.0.1:8050/
```

Available profiles include:

- `SIS18extraction` for standard SIS18 slow extraction;
- `SIS18extraction_mixed_beam` for two-ion tracking;
- `SIS18extraction_biomed` for IC data from the medical cave.

## Multi-GPU tracking

`track_multigpu()` divides an `xtrack.Particles` object between the available OpenCL GPU devices. Each worker constructs its own line and tracking context.

```python
from toolbox import track_multigpu

tracked_particles = track_multigpu(
    particles,
    line_constructor = build_line,
    num_turns = 100_000,
    num_gpus = 2,
    verbose = 1,
)
```

Because multiprocessing uses the `spawn` method, `line_constructor` must be a top-level, picklable function. Calls from scripts should be protected with:

```python
if __name__ == "__main__":
    ...
```

Particle-monitor snapshots can optionally be written to HDF5 by providing `record_every` and `monitor_output_directory`.

## Phase-space and separatrix tools

The package provides utilities for:

- locating the stable-region boundary;
- constructing stable and unstable particles;
- estimating the three vertices of a third-integer separatrix;
- reducing particle-monitor output to binned phase-space snapshots;
- combining snapshots from several GPU workers.

```python
from toolbox import PhaseSpaceSnapshots

snapshots = PhaseSpaceSnapshots(
    xlim = [-0.1, 0.1],
    pxlim = [-0.01, 0.01],
    n_bins = 100,
    filename = "particle_monitor_device.h5",
)

snapshots.save_data("phase_space_snapshots.h5")
```

## License

The package metadata declares the project under the MIT License. A corresponding `LICENSE` file should be added to the repository.
