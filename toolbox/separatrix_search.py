from typing import List
import numpy as np
from tqdm.notebook import tqdm
from xtrack import Line, Particles

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

		current_width = test_range[1] - test_range[0]

		progress_current = min(1, get_progress(test_range[1] - test_range[0])) * 100 # in %
		
		if with_progress:
			progress.update(progress_current - progress_init)  # Keep bar moving

		progress_init = progress_current
		
	if with_progress:
		progress.close()
		
	return test_range

