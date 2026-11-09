# Separatrix properties matching

An example of adjusting the spiral step and extraction angle by varying the sextupoles' phase and amplitude.

The lattice used is SIS18, available at Gitlab in json format. 
In this example, multiple isolated beamlines are created with `LineWorkerPool` for the parameters scan with differential evolution algorithm.

```python
import xtrack as xt
import numpy as np
import time
from scipy.optimize import differential_evolution
import toolbox as tb
from toolbox.parallel import LineWorkerPool, get_worker_line


e0 = 931.49410242e6
a_e = 5.48579909065e-4

C_ion = {
	'charge': +3,
	'A': 12.0,
	'mass_in_u': 12.0 - 3 * a_e,
	'mass': (12.0 - 3 * a_e) * e0, # in eV
	'kinetic_energy': 225e6 * (12.0 - 3 * a_e), # in eV
	'gamma': 1 + 225e6 / e0,
	'energy': (12.0 - 3 * a_e) * ( e0 + 225e6), # in eV
	'pdg_id': 1000060120
}

def construct_line(
	sextupole_phase: float = 0.0,
	sextupole_amplitude: float = 0.0,
	sextupole_offset: float = 0.0,  
	es_bump_amplitude: float = 0.0,
	ms_bump_amplitude: float = 0.0,
	) -> xt.Line:
	"""
	This is the function that creates and returs the lattice.

	This is needed for the workers to do the tracking on multiple
	GPU contexts.
	"""
	line = xt.load("sis18ring_extraction.json")

	# settings for -0.0037 tune distance
	line.vars['kqf1'] = 0.3739220481649916
	line.vars['kqd'] = -0.35220202131755707

	carbon_beam_ref = xt.Particles(
		mass0 = C_ion['mass'], 
		q0 = C_ion['charge'], 
		kinetic_energy0 = C_ion['kinetic_energy']
	)
	line.particle_ref = carbon_beam_ref

	line.vars['ks.phase'] =  sextupole_phase
	line.vars['ks.amp'] = sextupole_amplitude
	line.vars['ks.offset'] = sextupole_offset
	
	line.vars['es_bump.amp'] = es_bump_amplitude
	line.vars['ms_bump.amp'] = ms_bump_amplitude
	
	# Estimation based on May 2025 chroamticity measurements
	k2l_bends = 0.04
	bends, __ = line.get_elements_of_type(xt.Bend)
	for bend in bends:
		bend.knl[2] = k2l_bends / 2
	return line

May_2025_machine_settings = dict(
	sextupole_phase = 40. / 180. * np.pi,
	sextupole_amplitude = 0.06,
	sextupole_offset = 0.155,
	es_bump_amplitude = -1.8e-2,
	ms_bump_amplitude = 7e-3,
)

def calculate_separatrix(parameters, optimization_config) -> dict:
	
	line = get_worker_line()
	if line is None:
		raise RuntimeError("Worker was not initialized")

	amplitude, phase = parameters
	
	line.vars["ks.phase"] = phase
	line.vars["ks.amp"] = amplitude

	_search_parameters = (
		"septum", "ex_norm", "test_range", "n_particles", 
		"num_turns", "ion", "precision", "shrinkage_strength", 
		"max_iterations", "twiss"
	)

	res = tb.get_separatrix_properties(
		line, 
		**{
			key: optimization_config[key]
			for key in _search_parameters
			if key in optimization_config
		},
		verbose = 0
	)
	return res

def objective(parameters, optimization_config) -> float:
	"""Evaluates the scalar loss"""
	try:
		properties = calculate_separatrix(parameters, optimization_config)
	except tb.SeparatrixSearchError:
		return optimization_config["failure_penalty"]

	def residual(key):
		return (properties[key] - optimization_config[f"target_{key}"]) / optimization_config[f"{key}_tolerance"]

	loss = residual("spiral_step")**2 + residual("angle")**2

	if not np.isfinite(loss):
		return optimization_config["failure_penalty"]

	return float(loss)

if __name__ == "__main__":
	optimization_config = {
		"septum": {
			"entrance": "gs04me1e_en_aper",
			"exit": "gs04me1e_ex_aper",
		},
		"ex_norm": 1.6e-5,
		"n_particles": 2,
		"test_range": [-3.0, 0.0],
		"precision": 1e-4,
		"num_turns": 500,

		"target_spiral_step": 3.0e-3,
		"spiral_step_tolerance": 2.0e-4,
		"target_angle": -8.0e-3,
		"angle_tolerance": 2.0e-4,
		"failure_penalty": 1e6,
	}

	num_processes = 18
	start = time.time()

	with LineWorkerPool(
		construct_line,
		line_factory_kwargs = May_2025_machine_settings,
		processes = num_processes,
	) as pool:
		result = differential_evolution(
			objective,
			args = (optimization_config, ),
			bounds = [
				(0.01, 0.1),
				(np.deg2rad(5.0), np.deg2rad(280.0)),				
			],

			popsize = 9,
			maxiter = 30,

			workers = pool.map,
			updating = "deferred",

			strategy = "best1bin",
			mutation = (0.5, 1.0),
			recombination = 0.7,

			tol = 1e-2,
			disp = True,

			polish = False
		)

		final_properties = pool.apply(
			calculate_separatrix,
			args = (
				result.x,
				optimization_config
			),
		)
		print(f"Finished, elapsed time = {time.time() - start} s")

		print("Optimization successful:", result.success)
		print("Message:", result.message)
		print("Objective value:", result.fun)

		print(
			"Phase:",
			np.rad2deg(result.x[1]),
			"degrees",
		)

		print("Amplitude:", result.x[0])
		print("Properties:", final_properties)
```