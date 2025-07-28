In this repository I keep a package with some extra tools I commonly use for the analysis/tracking/etc. of the lattice in `xsuite`.

## Installation
Clone the repository and install the package with `pip`:
```bash
git clone https://github.com/drozzoff/toolbox.git
cd toolbox
pip3 install .
```
### Tools currently available
- Create and set an `xt.Line` of SIS18 synchrotron at the extraction with **`toolbox.set_sis18ring()`**. 

One has to provide a reference particle, eg:
```python
C_ion = {
	'charge': +3,
	'A': 12.0,
	'mass_in_u': 12.0 - 3 * a_e,
	'mass': (12.0 - 3 * a_e) * e0, # in eV
	'kinetic_energy': 225e6 * (12.0 - 3 * a_e), # in eV
	'gamma': 1 + 225e6 / e0,
	'energy': (12.0 - 3 * a_e) * ( e0 + 225e6), # in eV
	'pdg_id': 1000060120 # just the nuclei
}
```
and optionally, a working point (`qx` and/or `qy`), sextupolar component of the bends (`ksl_bends`), and setup for the RF exciter.
An example:
```python
import toolbox as tb

sis18ring = tb.set_sis18ring(
    reference_particle = C_ion,
    k2l_bends = 0.04,
    exciter_setup = tb.exc_bpsk(
        center_tune = 4 + 1/3. - 0.0037,
        tune_bandwidth = 0.01,
        timestep = 1e-8, # s
        excitation_duration = 2e-2, #s
        k0l = 1e-6,
        start_turn = 165,
        save_for_debug = True
    )
)
```
*Atm, requires 3 `MAD-X` files: `"SIS18RING_xtrack.seq"`, `"SIS18_TAU_1.000.str"`, and `"SIS18_cryocatchers_new.str"`.* Will create a separate repository with a plain **SIS18** lattice in `json` format later.
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

To be able ti use it, one has create a callback function for the `xsuite` tracking, which will send the data to the dashboard.

The data sent must correspond to the requested data field in `TrackingDashboard`.

```python
import json
import numpy as np
import socket

def callback(line, particles, **kwargs):
    """
    This is the callback function that will be called at each turn.
    Is used to send the data to the Dash server.
    """
    global current_turn
    n_alive_particles = int(sum(particles.state))
    
    after_excitor_is_on = particles.at_turn > 100
    at_last_turn = particles.at_turn == current_turn - 1
    
    at_last_turn_after_excitor_is_on = np.logical_and(at_last_turn, after_excitor_is_on)
    
    lost = particles.state == 0
    lost_properly = np.logical_and(lost, particles.x <= -0.055)
    at_start = abs(particles.s) < 1e-7
    at_septum_end = particles.s == 1.5
    
    lost_properly_after_exc = np.logical_and(lost_properly, at_last_turn_after_excitor_is_on)
    
    lost_last_turn_at_start = np.logical_and(lost_properly_after_exc, at_start)
    lost_last_turn_at_septum = np.logical_and(lost_properly_after_exc, at_septum_end)

    # identifying the ion types
    chi = particles.chi
    
    ion_type = np.full_like(chi, fill_value = "uidentified", dtype = object)
    
    ion_type[chi == 1.] = "carbon"
    ion_type[np.abs(chi - 0.99934958) < 1e-8] = "helium"
    
    data = {
        'turn': [current_turn], 
        'Nparticles': [n_alive_particles],
        'x_extracted_at_ES': particles.x[lost_last_turn_at_start].tolist(),
        'px_extracted_at_ES': particles.px[lost_last_turn_at_start].tolist(),
        'ES_septum_anode_loss_outside': [int(sum(lost_last_turn_at_septum))],
        "ion": ion_type[lost_last_turn_at_start].tolist()
    }
    sock.sendall((json.dumps(data) + '\n').encode())
    
    current_turn += 1
    return n_alive_particles


sis18ring.track(
	p, 
	num_turns = 1000,
	with_progress = True,
	turn_by_turn_monitor = True,
	log = xt.Log(callback = callback)
)
```

- Functions to find separatix and other phase space tools.

- Some plotting tools.