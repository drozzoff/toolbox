import numpy as np
from tqdm.notebook import tqdm
import xtrack as xt


def stable_particle_id(line: xt.Line, particles: xt.Particles, num_turns, **kwargs):
	"""
	Evaluates the `particle_id` of a particle in `particles` that is located
	in a stable region.
	
	The function assumes `particles` are generated with a continous set of `x` from
	negative values towards 0.
	
	Unstable particles are at the beginning, stable particles at the end

	Parameters
	----------
	line
		A beamline
	Particless
		Particles objec with `x` coordinate being continous
	num_turns
		Number of turns for the tracking

	Additional parameters
	---------------------
	verbose : int
		If <= 1, function does not generate any ouput. For `verbose > 1` plots the particles' tracks
	"""
	n_particles = particles._num_active_particles
	tw = line.twiss4d()

	verbose = kwargs.get("verbose", 0)

	if verbose > 1:
		import matplotlib.pyplot as plt
		import seaborn as sns

	line.track(particles, num_turns = num_turns, turn_by_turn_monitor = True)

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
	if verbose > 1:
		sns.set_style("darkgrid")
		fig, axes = plt.subplots()
		for i in range(n_particles):
			sns.scatterplot(
				x = x_norms[i],
				y = px_norms[i],
				color = "red", 
				alpha = 0.6, 
				s = 5,
				ax = axes
			)    
		plt.show()
	
	return particle_id_stable, jx_diff

def get_stable_limit(
	line: xt.Line, 
	ex_norm: float, 
	n_particles: int, 
	num_turns: int,
	*,
	test_range: list[float] = [-3.0, 0.0], 
	ion: xt.Particles | None = None, 
	precision: float = 1e-6,
	shrinkage_strength: int | str = "max",
	max_iterations: int = 2000,
	**kwargs
	):

	"""
	Find the normalized coordinates `x_norm` that produces stable and unstable
	trajectories in the phase space with the given precision.

	Parameters
	----------
	line
		beamline to do the tracking on
	ex_norm
		horizontal normalized emittance
	n_particles
		number of particles to use for the search
	num_turns
		number of turns to track for the search
	test_range
		initial guess for the location of the stable region limit. Must contain two elements both negative.
		The number corresponds to the normalized x coordinate to a sigma. (`xt.line.build_particles()`))
	ion
		A particle to use for the tracking.
		Default is `None` -> uses reference particle associated with a beamline.
	precision
		distance between the stable and unstable track. `test_range` is used to evaluate this.
	shrinkage_strength: int or str
		Defines how the `test_range` is shrinked in iterations.
		If `int` it leaves certain number of particles on the left and right of the identified region.
		If `str` is `'max'` no extra steps are added.
		Default is `'max'`.
	max_iterations : int
		Limit on the number of iterations. Default is `2000`.
		
	Additional parameters
	---------------------
	verbose : int
		If `verbose < 1` no output is printed.
		If `verbose == 1` prints the progress bar (**default**)
		If `verbose > 1` prints the progress bar and particles distribution at each iteration
	zeta : np.array
		`zeta` coordinate for the test beam
	delta : np.array
		`delta` coordinate for the test beam
	
	Returns
	-------
	List[float]
		First element is `x_norm` coordinate of the unstable particle. 
		Second element is `x_norm` coordinate stable particle.
	dict
		Contains the data of the iterations. Keys are `jx_diff` and `x_norm`.
	"""
	verbose = kwargs.get("verbose", 1)
	with_progress = verbose > 0
	progress = tqdm(total = 100, desc = "Tracking to find a stable region", leave = True) if with_progress else None
	
	get_progress = lambda x: 1 - (x - precision) / x
	progress_init = min(1, get_progress(test_range[1] - test_range[0])) * 100 # in %
	
	if with_progress:
		progress.update(progress_init)

	iterations_data = {
		'jx_diff': [],
		'x_norm': []
	}
	iter_id = 0
	while (test_range[1] - test_range[0]) > precision and iter_id < max_iterations:
		
		x_test = np.linspace(test_range[0], test_range[1], n_particles)
		iterations_data['x_norm'].append(x_test)
		
		test_beam = line.build_particles(
			x_norm = x_test,
			zeta = kwargs.get("zeta", 0.0),
			delta = kwargs.get("delta", 0.0),
			nemitt_x = ex_norm,
			mode = "normalized_transverse",
			method = '4d'
		)
		if isinstance(ion, xt.Particles):
			test_beam2 = xt.Particles(
				mass0 = line.particle_ref.mass0, 
				q0 = line.particle_ref.q0,
				gamma0 = line.particle_ref.gamma0, # we assume rigidity is the same as of the reference beam
				mass_ratio = ion.mass0 / line.particle_ref.mass0,
				charge_ratio = ion.q0 / line.particle_ref.q0,
				x = test_beam.x,
				px = test_beam.px,
				zeta = kwargs.get("zeta", 0.0),
				delta = kwargs.get("delta", 0.0)
			)
			test_beam = test_beam2

		particle_id_stable, jx_diff = stable_particle_id(
			line = line, 
			particles = test_beam, 
			num_turns = num_turns,
			verbose = verbose
		)
		iterations_data['jx_diff'].append(jx_diff)
		
		if shrinkage_strength == "max":
			test_range[0] = x_test[particle_id_stable - 1]
			test_range[1] = x_test[particle_id_stable]
		elif isinstance(shrinkage_strength, int):
			left = particle_id_stable - 1 - shrinkage_strength
			right = particle_id_stable + shrinkage_strength
			test_range[0] = x_test[left if left >= 0 else 0]
			test_range[1] = x_test[right if right < n_particles else n_particles - 1]
		
		if with_progress:
			progress_current = min(1, get_progress(test_range[1] - test_range[0])) * 100 # in %
			progress.update(progress_current - progress_init) 
	
			progress_init = progress_current

		iter_id += 1
	
	if with_progress:
		progress.close()
		
	return test_range, iterations_data

def get_stable_and_unstable_particle(
	line: xt.Line, 
	ex_norm: float, 
	n_particles: int, 
	num_turns: int,
	*,
	test_range: list[float] = [-3.0, 0.0], 
	ion: xt.Particles | None = None, 
	precision: float = 1e-6,
	shrinkage_strength: int | str = "max",
	max_iterations: int = 2000,
	**kwargs
	) -> tuple[tuple[xt.Particles, xt.Particles], dict]:
	"""
	Estimates the separatrix limit around 3rd integer resonance and returns 1 particle at
	stable trajectory, and one at the unstable trajectory.

	Parameters
	----------
	line
		beamline to do the tracking on
	ex_norm
		horizontal normalized emittance
	n_particles
		number of particles to use for the search
	num_turns
		number of turns to track for the search
	test_range
		initial guess for the location of the stable region limit. Must contain two elements both negative.
		The number corresponds to the normalized x coordinate to a sigma. (`xt.line.build_particles()`))
	ion
		A particle to use for the tracking. Default is `None` -> uses reference particle associated with a beamline.
	precision
		distance between the stable and unstable track. `test_range` is used to evaluate this.
	shrinkage_strength: int or str
		Defines how the `test_range` is shrinked in iterations.
		If `int` it leaves certain number of particles on the left and right of the identified
		If `str` is `'max'` no extra steps are added.
		Default is `'max'`.
	max_iterations : int
		Limit on the number of iterations. Default is `2000`.
		
	Additional parameters
	---------------------
	
	verbose : int
		If `verbose < 1` no output is printed.
		If `verbose == 1` prints the progress bar (**default**)
		If `verbose > 1` prints the progress bar and particles distribution at each iteration
	zeta : np.array
		`zeta` coordinate for the test beam
	delta : np.array
		`delta` coordinate for the test beam

	Returns
	-------
	List
		First element is a stable particle. 
		Second particle is unstable particle.
	dict
		Contains the data of the iterations. Keys are `jx_diff` and `x_norm`.
	"""
	res, iteration_data = get_stable_limit(
		line, 
		ex_norm, 
		n_particles,
		num_turns,
		test_range = test_range,
		ion = ion,
		precision = precision,
		shrinkage_strength = shrinkage_strength,
		max_iterations = max_iterations
		**kwargs
	)

	tmp = line.build_particles(
		x_norm = res,
		nemitt_x = ex_norm,
		mode = "normalized_transverse",
		method = "4d"
	)

	mass_ratio = 1.0 if ion is None else ion.mass0 / line.particle_ref.mass0
	charge_ratio = 1.0 if ion is None else ion.q0 / line.particle_ref.q0

	stable_particle = xt.Particles(
		mass0 = line.particle_ref.mass0, 
		q0 = line.particle_ref.q0,
		gamma0 = line.particle_ref.gamma0,
		mass_ratio = mass_ratio,
		charge_ratio = charge_ratio,
		x = tmp.x[1],
		px = tmp.px[1],
		delta = kwargs.get('delta', 0),
		zeta = kwargs.get('zeta', 0)
	)

	unstable_particle = xt.Particles(
		mass0 = line.particle_ref.mass0,
		q0 = line.particle_ref.q0,
		gamma0 = line.particle_ref.gamma0,
		mass_ratio = mass_ratio,
		charge_ratio = charge_ratio,
		x = tmp.x[0],
		px = tmp.px[0],
		delta = kwargs.get('delta', 0),
		zeta = kwargs.get('zeta', 0)
	)

	return (stable_particle, unstable_particle), iteration_data
