In this repository I keep a package with some extra tools I commonly use for the analysis/tracking/etc. of the lattice in `xsuite`.

## Installation
Clone the repository and install the package with `pip`:
```bash
git clone https://github.com/drozzoff/toolbox.git
cd toolbox
pip3 install .
```
### Tools available
- Simple tool to split the tracking over multiple GPUs with **`toolbox.track_multigpu()`**.

- Build `xt.Exciter` with BPSK or with frequency chirp.

- Analyze the tune spectrogram with **`toolbox.parse_tunes()`**.

It looks for a region to search, fit every timestep with a Gaussian, and filters ou bad fits.
It assumes only 1 spectral line is present and does not drift very much. There are multiple parameters, check the function documentation for details.
```python
import toolbox as tb

p_0 = pd.read_pickle(f"spectrogram.pkl")

res = tb.parse_tunes(
    data = p_0,
    noise_percentile = 0.99,
    search_region_width = 5,
    noise_width = 3,
    peak_min_length = 1,
    post_dilation = 3,
    weights_func = tb.rational_weight,
    r2_score_cut = 0.5,
    verbose = 2
)
```

- Extraction dahsboard with **`toolbox.TrackingDashboard`**.

Sets up a `Dash` dashboard and visualizes the requested data from the `xsuite` tracking.

An example:
```python
import toolbox as tb

test = tb.TrackingDashboard(
	port = 35235, 
	data_to_monitor = [
		"intensity", 
		"ES_septum_anode_losses",
#			"spill",
		"spill_mixed",
		"ES_entrance_phase_space",
#			"MS_entrance_phase_space",
		"separatrix"
	]
)
test.start_listener()

test.run_dash_server()
```
It listens at the port `35235` and can be accessed at `http://127.0.0.1:8050/`.

To be able to use it, one has create a callback function for the `xsuite` tracking, which will send the data to the dashboard.

The data sent must correspond to the requested data field in `TrackingDashboard`.

- Functions to find separatix and other phase space tools.

- Some plotting tools.