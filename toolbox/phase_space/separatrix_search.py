import numpy as np
from tqdm.notebook import tqdm
import xtrack as xt


def stable_particle_id(
	line: xt.Line, 
	particles: xt.Particles, 
	num_turns,
	twiss: xt.TwissTable | None = None,
	**kwargs
	) -> tuple[int, list]:
	"""
	Evaluates the `particle_id` of a particle in `particles` that is located
	in a stable region.
	
	The function assumes `particles` are generated with a continous set of `x` from
	negative values towards 0.
	
	Unstable particles are at the beginning, stable particles at the end.

	Parameters
	----------
	line
		A beamline
	Particless
		Particles objec with `x` coordinate being continous
	num_turns
		Number of turns for the tracking
	twiss
		TwissTable for the lattice. If not provided, is evaluated internally.
	
	Additional parameters
	---------------------
	verbose : int
		If <= 1, function does not generate any ouput. 
		For `verbose > 1` plots the particles' tracks

	Returns
	-------
	Tuple with the id of a stable particle and list of action differences
	extracted from the tracking
	"""
	n_particles = particles._num_active_particles

	if isinstance(twiss, xt.TwissTable):
		tw = twiss
	else:
		tw = line.twiss4d()	

	verbose = kwargs.get("verbose", 0)

	line.track(
		particles, 
		num_turns = num_turns, 
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
		Jx = list(map(
			lambda x, y: np.sqrt(x**2 + y**2), 
			x_norms[particle_id][:lost_at_turn + 1], 
			px_norms[particle_id][:lost_at_turn + 1]
		))
	  
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
		import matplotlib.pyplot as plt
		import seaborn as sns

		with sns.axes_style("whitegrid"):
			for i in range(n_particles):
				sns.scatterplot(
					x = x_norms[i],
					y = px_norms[i],
					color = "red", 
					alpha = 0.6, 
					s = 5,
				)

			plt.xlabel("X")
			plt.ylabel("PX")

			plt.show()
	
	return particle_id_stable, jx_diff

def get_stable_limit(
	line: xt.Line, 
	ex_norm: float, 
	n_particles: int = 21, 
	num_turns: int = 1_000,
	*,
	test_range: list[float] = [-3.0, 0.0], 
	ion: xt.Particles | None = None, 
	precision: float = 1e-4,
	shrinkage_strength: int | str = "max",
	max_iterations: int = 2_000,
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
		initial guess for the location of the stable region limit. Must contain 
		two elements both negative. The number corresponds to the normalized 
		x coordinate to a sigma. (`xt.line.build_particles()`))
	ion
		A particle to use for the tracking.
		Default is `None` -> uses reference particle associated with a beamline.
	precision
		Distance between the stable and unstable track. The distance is 
		in the normalised phase space divided by sigma.
	shrinkage_strength: int or str
		Defines how the `test_range` is shrinked in iterations.
		If `int` it leaves certain number of particles on the left and 
		right of the identified region.
		If `str` is `'max'` no extra steps are added.
		Default is `'max'`.
	max_iterations : int
		Limit on the number of iterations. Default is `2000`.
		
	Additional parameters
	---------------------
	verbose : int
		If `verbose < 1` no output is printed.
		If `verbose == 1` prints the progress bar (**default**)
		If `verbose > 1` prints the progress bar and particles distribution 
		at each iteration

	Returns
	-------
	List[float]
		First element is `x_norm` coordinate of the unstable particle. 
		Second element is `x_norm` coordinate stable particle.
	dict
		Contains the data of the iterations. Keys are `jx_diff` and `x_norm`.
	"""
	if precision <= 0:
		raise ValueError("precision must be positive")

	verbose = kwargs.get("verbose", 1)
	range_low = test_range[0]
	range_high = test_range[1]
	initial_width = range_high - range_low

	with_progress = verbose > 0
	if with_progress:
		log_span = np.log(initial_width / precision)

		def convergence_percent(width):
			if width <= precision:
				return 100.0

			completed = np.log(initial_width / width) / log_span
			return 100.0 * np.clip(completed, 0.0, 1.0)
		
		progress = tqdm(
			total = 100, 
			desc = "Tracking to find a stable region", 
	    	leave = True
		)

	get_progress = lambda x: 1 - (x - precision) / x
	progress_init = min(1, get_progress(range_high - range_low)) * 100 # in %
	
	if with_progress:
		progress.update(progress_init)

	iterations_data = {
		'jx_diff': [],
		'x_norm': []
	}

	iter_id = 0
	while (range_high - range_low) > precision and iter_id < max_iterations:
		
		x_test = np.linspace(range_low, range_high, n_particles)
		iterations_data['x_norm'].append(x_test)
		
		test_beam = line.build_particles(
			x_norm = x_test,
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
				y = ion.y,
				py = ion.py,
				zeta = ion.zeta,
				delta = ion.delta
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
			range_low = x_test[particle_id_stable - 1]
			range_high = x_test[particle_id_stable]
		elif isinstance(shrinkage_strength, int):
			left = particle_id_stable - 1 - shrinkage_strength
			right = particle_id_stable + shrinkage_strength
			range_low = x_test[left if left >= 0 else 0]
			range_high = x_test[right if right < n_particles else n_particles - 1]
		
		if with_progress:
			current_width = range_high - range_low
			progress_current = convergence_percent(current_width)
			progress.update(progress_current - progress.n) 
	
			progress.set_postfix(
				width=f"{current_width:.2e}",
				target=f"{precision:.2e}",
				iteration=iter_id + 1,
			)

		iter_id += 1
	
	if with_progress:
		progress.close()
		
	return [range_low, range_high], iterations_data

def get_stable_and_unstable_particle(
	line: xt.Line, 
	ex_norm: float, 
	n_particles: int = 21, 
	num_turns: int = 1_000,
	*,
	test_range: list[float] = [-3.0, 0.0], 
	ion: xt.Particles | None = None, 
	precision: float = 1e-6,
	shrinkage_strength: int | str = "max",
	max_iterations: int = 2_000,
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
		max_iterations = max_iterations,
		**kwargs
	)

	tmp = line.build_particles(
		x_norm = res,
		nemitt_x = ex_norm,
		mode = "normalized_transverse",
		method = "4d"
	)

	beam_creation_opt = {}
	if ion is not None:
		beam_creation_opt['mass_ratio'] = ion.mass0 / line.particle_ref.mass0
		beam_creation_opt['charge_ratio'] = ion.q0 / line.particle_ref.q0
		beam_creation_opt['y'] = ion.y
		beam_creation_opt['py'] = ion.py
		beam_creation_opt['delta'] = ion.delta
		beam_creation_opt['zeta'] = ion.zeta


	stable_particle = xt.Particles(
		mass0 = line.particle_ref.mass0, 
		q0 = line.particle_ref.q0,
		gamma0 = line.particle_ref.gamma0,
		x = tmp.x[1],
		px = tmp.px[1],
		**beam_creation_opt
	)

	unstable_particle = xt.Particles(
		mass0 = line.particle_ref.mass0,
		q0 = line.particle_ref.q0,
		gamma0 = line.particle_ref.gamma0,
		x = tmp.x[0],
		px = tmp.px[0],
		**beam_creation_opt
	)

	return (stable_particle, unstable_particle), iteration_data
