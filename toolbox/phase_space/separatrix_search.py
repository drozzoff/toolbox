import numpy as np
from tqdm.notebook import tqdm
import xtrack as xt
from toolbox.phase_space.exceptions import NoStableParticleError, NoUnstableParticleError


def stable_particle_id_by_jx_diff(
	line: xt.Line, 
	particles: xt.Particles, 
	num_turns: int,
	*,
	twiss: xt.TwissTable | None = None,
	verbose: int = 1
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
	twiss : xt.TwissTable
		TwissTable for the lattice. If not provided, is evaluated internally.
	verbose : int
		If <= 1, function does not generate any ouput. 
		For `verbose > 1` plots the particles' tracks
		
	Returns
	-------
	Tuple with the id of a stable particle and list of action differences
	extracted from the tracking
	"""
	n_particles = particles._num_active_particles

	if twiss is None:
		twiss = line.twiss4d()

	line.track(
		particles, 
		num_turns = num_turns, 
		turn_by_turn_monitor = True
	)

	rec = line.record_last_track
	nc = twiss.get_normalized_coordinates(rec.data)

	x_norms = np.asarray(nc.x_norm).reshape(n_particles, -1)
	px_norms = np.asarray(nc.px_norm).reshape(n_particles, -1)
	radius = np.hypot(x_norms, px_norms)

	particles.sort(interleave_lost_particles = True)	
	
	turns = np.arange(x_norms.shape[1])
	mask = turns[None, :] <= particles.at_turn[:, None]
	
	Jx_mean = np.sum(radius, axis = 1, where = mask) / mask.sum(axis = 1)
	jx_diff = np.concatenate((
		np.zeros(1, dtype = Jx_mean.dtype),
		np.abs(np.diff(Jx_mean)),
	))
	
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
	
	return np.arange(n_particles) >= particle_id_stable, jx_diff

def stable_particle_id_by_loss(
	line: xt.Line, 
	particles: xt.Particles, 
	num_turns,
	*,
	verbose: int = 1,
	twiss: xt.TwissTable | None = None,
	) -> np.array:
	line.track(
		particles, 
		num_turns = num_turns, 
		turn_by_turn_monitor = verbose > 1
	)

	if verbose > 1:
		import matplotlib.pyplot as plt
		import seaborn as sns

		if twiss is None:
			twiss = line.twiss4d()

		n_particles = particles._capacity

		rec = line.record_last_track
		nc = twiss.get_normalized_coordinates(rec.data)
	
		x_norms = np.asarray(nc.x_norm).reshape(n_particles, -1)
		px_norms = np.asarray(nc.px_norm).reshape(n_particles, -1)

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
	
	particles.sort(interleave_lost_particles = True)	
	
	return particles.state == 1

def get_stable_limit(
	line: xt.Line, 
	ex_norm: float, 
	n_particles: int = 2, 
	num_turns: int = 500,
	*,
	test_range: list[float] = [-3.0, 0.0], 
	ion: xt.Particles | None = None, 
	precision: float = 1e-4,
	shrinkage_strength: int | str = "max",
	max_iterations: int = 2_000,
	twiss: xt.TwissTable | None = None,
	verbose: int = 1,
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
	twiss : xt.TwissTable
		TwissTable for the lattice. If not provided, is evaluated internally.
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

	if twiss is None:
		twiss = line.twiss4d()

	binary_search = n_particles == 1

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

	def update_progress(current_width):
		progress_current = convergence_percent(current_width)
		progress.update(progress_current - progress.n) 

		progress.set_postfix(
			width = f"{current_width:.2e}",
			target = f"{precision:.2e}",
			iteration = iter_id + 1,
		)

	W0 = twiss.W_matrix[0]
	particle_on_co = twiss.particle_on_co
	iter_id = 0
	while (range_high - range_low) > precision and iter_id < max_iterations:

		if binary_search:
			if iter_id == 0:
				x_test = np.array([range_low, range_high])
			else:
				x_test = np.array([0.5 * (range_low + range_high)])
		else:
			if iter_id == 0:
				x_test = np.linspace(range_low, range_high, n_particles)
			else:
				x_test = np.linspace(range_low, range_high, n_particles + 2)[1:-1]
			
		test_beam = line.build_particles(
			x_norm = x_test,
			nemitt_x = ex_norm,
			mode = "normalized_transverse",
			method = '4d',
			W_matrix = W0,
			particle_on_co = particle_on_co
		)
		if isinstance(ion, xt.Particles):
			test_beam2 = xt.Particles(
				mass0 = line.particle_ref.mass0, 
				q0 = line.particle_ref.q0,
				gamma0 = line.particle_ref.gamma0,
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

		stable_mask = stable_particle_id_by_loss(
			line = line, 
			particles = test_beam, 
			num_turns = num_turns,
			verbose = verbose,
			twiss = twiss
		)

		stable_indices = np.flatnonzero(stable_mask)

		if stable_indices.size == 0:
			if iter_id == 0:
				raise NoStableParticleError(
					"No stable particle was found in the initial test range."
				)

			# No stable particles, looking on the right side
			range_low = x_test[-1]
			range_high = range_high

			if with_progress:
				update_progress(range_high - range_low)

			iter_id += 1
			continue

		if stable_indices.size == x_test.size:
			if iter_id == 0:
				raise NoUnstableParticleError(
					"No unstable particle was found in the initial test range."
				)
		
			# All particles are stable, looking on the left side
			range_low = range_low
			range_high = x_test[0]

			if with_progress:
				update_progress(range_high - range_low)

			iter_id += 1
			continue

		particle_id_stable = int(stable_indices[0])

		if not binary_search:
			if shrinkage_strength == "max":
				range_low = x_test[particle_id_stable - 1]
				range_high = x_test[particle_id_stable]
			elif isinstance(shrinkage_strength, int):
				left = particle_id_stable - 1 - shrinkage_strength
				right = particle_id_stable + shrinkage_strength
				range_low = x_test[left if left >= 0 else 0]
				range_high = x_test[right if right < n_particles else n_particles - 1]
		
		if with_progress:
			update_progress(range_high - range_low)

		iter_id += 1
	
	if with_progress:
		progress.close()
		
	return [range_low, range_high], None

def get_stable_and_unstable_particle(
	line: xt.Line, 
	ex_norm: float, 
	n_particles: int = 2, 
	num_turns: int = 500,
	*,
	test_range: list[float] = [-3.0, 0.0], 
	ion: xt.Particles | None = None, 
	precision: float = 1e-6,
	shrinkage_strength: int | str = "max",
	max_iterations: int = 2_000,
	twiss: xt.TwissTable | None = None,
	verbose: int = 1,
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
		initial guess for the location of the stable region limit. 
		Must contain two elements both negative. The number corresponds to the 
		normalized x coordinate to a sigma. (`xt.line.build_particles()`))
	ion
		A particle to use for the tracking. Default is `None` -> uses reference
		particle associated with a beamline.
	precision
		distance between the stable and unstable track. `test_range` is used 
		to evaluate this.
	shrinkage_strength: int or str
		Defines how the `test_range` is shrinked in iterations.
		If `int` it leaves certain number of particles on the left and right 
		of the identified
		If `str` is `'max'` no extra steps are added.
		Default is `'max'`.
	max_iterations : int
		Limit on the number of iterations. Default is `2000`.
	twiss : xt.TwissTable
		TwissTable for the lattice. If not provided, is evaluated internally.
	verbose : int
		If `verbose < 1` no output is printed.
		If `verbose == 1` prints the progress bar (**default**)
		If `verbose > 1` prints the progress bar and particles distribution 
		at each iteration

	Returns
	-------
	List
		First element is a stable particle. 
		Second particle is unstable particle.
	dict
		Contains the data of the iterations. Keys are `jx_diff` and `x_norm`.
	"""
	if twiss is None:
		twiss = line.twiss4d()

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
		twiss = twiss,
		verbose = verbose,
	)

	W0 = twiss.W_matrix[0]
	tmp = line.build_particles(
		x_norm = res,
		nemitt_x = ex_norm,
		mode = "normalized_transverse",
		W_matrix = W0,
		particle_on_co = twiss.particle_on_co,
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
