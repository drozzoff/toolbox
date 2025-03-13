from typing import List, Tuple
import numpy as np
from tqdm.notebook import tqdm
from xtrack import Line, Particles, ParticlesMonitor
from pandas import DataFrame

def _stable_particle_id(line: Line, particles: Particles, n_particles: int, **kwargs):
	"""
	Evaluates the `particle_id` of a particle in `particles` that is located
	in a stable region.

	The function assumes `particles` are generated with a continous set of `x`.

	Unstable particles are at the beginning, stable particles at the end
	"""
	tw = line.twiss4d()
	
	line.track(
		particles,
		num_turns = kwargs.get("num_turns", 2000),
		turn_by_turn_monitor = True
	)
	
	rec = line.record_last_track
	nc = tw.get_normalized_coordinates(rec.data)
	
	x_norms = np.split(nc.x_norm, n_particles)
	px_norms = np.split(nc.px_norm, n_particles)

	Jx_mean = []

	particles.sort(interleave_lost_particles = True)

	for particle_id in range(n_particles):
		lost_at_turn = particles.at_turn[particle_id]
		Jx = list(map(lambda x, y: np.sqrt(x**2 + y**2), x_norms[particle_id][:lost_at_turn + 1], px_norms[particle_id][:lost_at_turn + 1]))
	  
		Jx_mean.append(np.mean(Jx))

	jx_diff = []
	for i, var in enumerate(Jx_mean):
		if i == 0:
			jx_diff.append(0)
			continue
		else:
			jx_diff.append(abs(var - Jx_mean[i - 1]))
	
	particle_id_stable = np.argmax(jx_diff)
	
	return particle_id_stable, jx_diff



def get_stable_limit(line: Line, test_range: List[float], ex_norm: float, n_particles: int, num_turns: int, precision = 1e-6, **kwargs):
	"""
	Find a stable region of a given line.

	Parameters
	----------
	line
		beamline to do the tracking on
	test_range
		initial guess for thelocation of the stable region limit. Must contain two elements both negative.
		The number corresponds to the normalized x coordinate to a sigma. (`xt.line.build_particles()`))
	ex_norm
		horizontal normalized emittance
	n_particles
		number of particles to use for the search
	num_turns
		number of turns to track for the search
	precision
		distance between the stable and unstable track. `test_range` is used to evaluate this.
	
	Additiona parameters
	---------------------
	with_progress
		whether to show a progress bar or not. Default is `False`
	"""
	with_progress = kwargs.get('with_progress', False)
	
	progress = tqdm(total = 100, desc = "Tracking to find a stable region", leave = True) if with_progress else None

	get_progress = lambda x: 1 - (x - precision) / x
	progress_init = min(1, get_progress(test_range[1] - test_range[0])) * 100 # in %
	
	if with_progress:
		progress.update(progress_init)

	while (test_range[1] - test_range[0]) > precision:
		
		x_test = np.linspace(test_range[0], test_range[1], n_particles)
		
		test_particles = line.build_particles(
			x_norm = x_test,
			px_norm = 0.0,
			y_norm = 0.0,
			py_norm = 0.0,
			nemitt_x = ex_norm,
			nemitt_y = 0.0,
			zeta = 0.0,
			delta = 0.0,
			method = '4d'
		)
		
		particle_id_stable, __ = _stable_particle_id(
			line = line, 
			particles = test_particles, 
			n_particles = n_particles,
			num_turns = num_turns
		)
		
		test_range[0] = x_test[particle_id_stable - 1]
		test_range[1] = x_test[particle_id_stable]         

		if with_progress:
			progress_current = min(1, get_progress(test_range[1] - test_range[0])) * 100 # in %
			progress.update(progress_current - progress_init) 

		progress_init = progress_current
		
	if with_progress:
		progress.close()
		
	return test_range

def get_phase_portrait2d(monitor: ParticlesMonitor, particles: Particles, at_turn: int, plane: str = 'x') -> DataFrame:
	"""
	Get the phase space portrait of the particles at a given turn from a last track monitor (`.record_last_track`).
	If the particle was lost before `at_turn` returns the last recorded coordinate.

	Using custom `ParticlesMonitor` will fail, because it does have a structured data, eg. `monitor.x`.

	Parameters
	----------
	particles
		beam that was used for the tracking
	at_turn
		turn to get the phase space portrait at
	plane
		plane for the phase space portrait. Must be either 'x', 'y' or, 'z'

	Returns
	-------
	DataFrame
		phase space portrait
	"""
	particles.sort(interleave_lost_particles = True)

	# if particle is lost before `at_turn` returns the turn it was lost at.
	# Otherwise returns `at_turn`
	at_turn_flat = list(map(lambda x: at_turn if x > at_turn else x, particles.at_turn))
	
	res = {}
	
	if plane == 'x':
		res['x'] = np.array(list(map(lambda x, y: x[y], monitor.x, at_turn_flat)))
		res['px'] = np.array(list(map(lambda x, y: x[y], monitor.px, at_turn_flat)))
	elif plane == 'y':
		res['y'] = np.array(list(map(lambda x, y: x[y], monitor.y, at_turn_flat)))
		res['py'] = np.array(list(map(lambda x, y: x[y], monitor.py, at_turn_flat)))
	elif plane == 'z':
		res['zeta'] = np.array(list(map(lambda x, y: x[y], monitor.zeta, at_turn_flat)))
		res['delta'] = np.array(list(map(lambda x, y: x[y], monitor.delta, at_turn_flat)))
	else:
		raise ValueError("plane must be either 'x', 'y' or 'z'")

	return DataFrame(res)

def get_phase_portrait4d(monitor: ParticlesMonitor, particles: Particles, at_turn: int) -> DataFrame:
	"""
	Get the transverse phase space portrait of the particles at a given turn from a last track monitor (`.record_last_track`).
	If the particle was lost before `at_turn` returns the last recorded coordinate.

	Using custom `ParticlesMonitor` will fail, because it does have a structured data, eg. `monitor.x`.

	Parameters
	----------
	particles
		beam that was used for the tracking
	at_turn
		turn to get the phase space portrait at

	Returns
	-------
	DataFrame
		phase space portrait
	"""
	particles.sort(interleave_lost_particles = True)

	# if particle is lost before `at_turn` returns the turn it was lost at.
	# Otherwise returns `at_turn`
	at_turn_flat = list(map(lambda x: at_turn if x > at_turn else x, particles.at_turn))
	
	res = {}
	
	res['x'] = np.array(list(map(lambda x, y: x[y], monitor.x, at_turn_flat)))
	res['px'] = np.array(list(map(lambda x, y: x[y], monitor.px, at_turn_flat)))
	res['y'] = np.array(list(map(lambda x, y: x[y], monitor.y, at_turn_flat)))
	res['py'] = np.array(list(map(lambda x, y: x[y], monitor.py, at_turn_flat)))

	return DataFrame(res)

def get_phase_portrait6d(monitor: ParticlesMonitor, particles: Particles, at_turn: int) -> DataFrame:
	"""
	Get the transverse phase space portrait of the particles at a given turn from a last track monitor (`.record_last_track`).
	If the particle was lost before `at_turn` returns the last recorded coordinate.

	Using custom `ParticlesMonitor` will fail, because it does have a structured data, eg. `monitor.x`.

	Parameters
	----------
	particles
		beam that was used for the tracking
	at_turn
		turn to get the phase space portrait at

	Returns
	-------
	DataFrame
		phase space portrait
	"""
	particles.sort(interleave_lost_particles = True)

	# if particle is lost before `at_turn` returns the turn it was lost at.
	# Otherwise returns `at_turn`
	at_turn_flat = list(map(lambda x: at_turn if x > at_turn else x, particles.at_turn))
	
	res = {}
	
	res['x'] = np.array(list(map(lambda x, y: x[y], monitor.x, at_turn_flat)))
	res['px'] = np.array(list(map(lambda x, y: x[y], monitor.px, at_turn_flat)))
	res['y'] = np.array(list(map(lambda x, y: x[y], monitor.y, at_turn_flat)))
	res['py'] = np.array(list(map(lambda x, y: x[y], monitor.py, at_turn_flat)))
	res['zeta'] = np.array(list(map(lambda x, y: x[y], monitor.zeta, at_turn_flat)))
	res['delta'] = np.array(list(map(lambda x, y: x[y], monitor.delta, at_turn_flat)))

	return DataFrame(res)

def compute_simple_masks(particles: Particles):
	"""
	Compute the masks for the particles compatible for the use with the output of
	phase portrait functions.
	"""
	_min_offset = 1e-7

	reached_start = abs(particles.s) < _min_offset
	lost = lambda at_turn: particles.at_turn < at_turn

	res = {
		'reached_start': reached_start,
		'lost': lost,
		'lost_at_start': lambda at_turn: np.logical_and(lost(at_turn), reached_start),
		'lost_not_at_start': lambda at_turn: np.logical_and(lost(at_turn), ~reached_start)
	}

	return res

	