from typing import List, Tuple
import numpy as np
from tqdm.notebook import tqdm
from xtrack import Line, Particles, ParticlesMonitor
from pandas import DataFrame

def stable_particle_id(line: Line, particles: Particles, n_particles: int, **kwargs):
	"""
	Evaluates the `particle_id` of a particle in `particles` that is located
	in a stable region.
	
	The function assumes `particles` are generated with a continous set of `x`.
	
	Unstable particles are at the beginning, stable particles at the end
	"""
	tw = line.twiss4d()
	
	if kwargs.get("debug", False):
		import matplotlib.pyplot as plt
		import seaborn as sns

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
	if kwargs.get("debug", False):
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
		line: Line, 
		ion: dict, 
		test_range: List[float], 
		ex_norm: float, 
		n_particles: int, 
		num_turns: int, 
		precision: float = 1e-6, 
		**kwargs
	):

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
	
	Additional parameters
	---------------------
	shrinkage_strength: int or str
		Defines how the `test_range` is shrinked in iterations.
		If `int` adds extra steps to the left and right of the stable particle.
		If `str` is `'max'` no extra steps are added.
		Default is `'max'`.
	max_iterations : int
		Limit on the number of iterations. Default is `2000`.
	with_progress : bool
		whether to show a progress bar or not. Default is `False`

	Returns
	-------
	List[float]
		First element is unstable particle, second element is stable particle.
	dict
		Contains the data of the iterations. Keys are `jx_diff` and `x_norm`.
	"""
	with_progress = kwargs.get('with_progress', False)
	
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
	while (test_range[1] - test_range[0]) > precision and iter_id < kwargs.get('max_iterations', 2000):
		
		x_test = np.linspace(test_range[0], test_range[1], n_particles)
		iterations_data['x_norm'].append(x_test)
		
		test_beam = line.build_particles(
			x_norm = x_test,
			nemitt_x = ex_norm,
			mode = "normalized_transverse",
			method = '4d'
		)

		test_particles = Particles(
			mass0 = line.particle_ref.mass0, 
			q0 = line.particle_ref.q0,
			gamma0 = line.particle_ref.gamma0,
			mass_ratio = ion['mass'] / line.particle_ref.mass0,
			charge_ratio = ion['charge'] / line.particle_ref.q0,
			x = test_beam.x,
			px = test_beam.px,
			zeta = kwargs.get("zeta", 0.0),
			delta = kwargs.get("delta", 0.0)
		)
		particle_id_stable, jx_diff = stable_particle_id(
			line = line, 
			particles = test_particles, 
			n_particles = n_particles,
			num_turns = num_turns,
			debug = kwargs.get("debug", False)
		)
		iterations_data['jx_diff'].append(jx_diff)
		
		shrinkage_strength = kwargs.get("shrinkage_strength", "max")
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
		line: Line, 
		ion: dict,
		ex_norm: float,  
		delta: float,
		calculation_settings: dict,
		**kwargs
	) -> tuple[tuple[Particles, Particles], dict]:
	"""
	
	Additional paramters
	--------------------
	verbose

	"""
	
	res, iteration_data = get_stable_limit(
		line = line, 
		ion = ion, 
		ex_norm = ex_norm,
		delta = delta,
		with_progress = True if kwargs.get('verbose', 0) > 0 else False,
		debug = True if kwargs.get('verbose', 0) > 1 else False,
		**calculation_settings
	)
	
	# res contains the normalized coordinates in sigma of stable and unstable particles

	tmp = line.build_particles(
		x_norm = res,
		nemitt_x = ex_norm,
		mode = "normalized_transverse",
		method = "4d"
	)

	stable_particle = Particles(
		mass0 = line.particle_ref.mass0, 
		q0 = line.particle_ref.q0,
		gamma0 = line.particle_ref.gamma0,
		mass_ratio = ion['mass'] / line.particle_ref.mass0,
		charge_ratio = ion['charge'] / line.particle_ref.q0,
		x = tmp.x[1],
		px = tmp.px[1],
		delta = delta
	)

	unstable_particle = Particles(
		mass0 = line.particle_ref.mass0,
		q0 = line.particle_ref.q0,
		gamma0 = line.particle_ref.gamma0,
		mass_ratio = ion['mass'] / line.particle_ref.mass0,
		charge_ratio = ion['charge'] / line.particle_ref.q0,
		x = tmp.x[0],
		px = tmp.px[0],
		delta = delta
	)

	return (stable_particle, unstable_particle), iteration_data

def get_separatrix_vertices(
		line: Line, 
		ion: dict,
		ex_norm: float,  
		delta: float,
		ransac_settings: dict,
		calculation_settings: dict,
		**kwargs
	):
	"""
	Parameters
	----------
	ransac_settings 
		Parameter for RANSAC

	Additional parameters
	---------------------
	verbose

	"""
	from skimage.measure import ransac, LineModelND

	verbose = kwargs.get('verbose', 0)

	particles, __ = get_stable_and_unstable_particle(
		line = line,
		ion = ion,
		ex_norm = ex_norm,
		delta = delta,
		calculation_settings = calculation_settings,
		verbose = verbose
	)
	stable_particle, unstable_particle = particles

	p = stable_particle.copy()
	line.track(p, num_turns = calculation_settings['num_turns'], turn_by_turn_monitor = True)
	stable_rec = line.record_last_track

	p = unstable_particle.copy()
	line.track(p, num_turns = 1000, turn_by_turn_monitor = True)
	unstable_rec = line.record_last_track

	_zero_st = (stable_rec.x[0] == 0) & (stable_rec.px[0] == 0)
	_zero_unst = (unstable_rec.x[0] == 0) & (unstable_rec.px[0] == 0)

	separatrix = dict(
		x_stable = stable_rec.x[0][~_zero_st],
		px_stable = stable_rec.px[0][~_zero_st],
		
		x_unstable = unstable_rec.x[0][~_zero_unst],
		px_unstable = unstable_rec.px[0][~_zero_unst]
	)

	separatrix['x'] = np.concatenate((separatrix['x_stable'], separatrix['x_unstable']))
	separatrix['px'] = np.concatenate((separatrix['px_stable'], separatrix['px_unstable']))

	p = np.c_[separatrix['x_stable'], separatrix['px_stable']]

	lines_properties = []
	# using RANSAC to find 3 lines in the phase space
	for i in range(3):

		model, inliers = ransac(p, LineModelND, min_samples = 2, **ransac_settings)

		p0, u = model.params
		
		u = - u / np.linalg.norm(u)

		right_amp = (max(p[:, 0][inliers]) - p0[0]) / u[0]
		left_amp = (min(p[:, 0][inliers]) - p0[0]) / u[0]
		
		L = np.array([p0 + u * left_amp, p0 + u * right_amp])
		a = (L[1, 1] - L[0, 1]) / (L[1, 0] - L[0, 0])
		b = L[0, 1] - a * L[0, 0]

		lines_properties.append(dict(a = a, b = b))

		if verbose == 2:
			import matplotlib.pyplot as plt
			import seaborn as sns
			data = DataFrame({
				'x': p[:, 0],
				'px': p[:, 1],
				'group': np.where(inliers, "Inliers", "Outliers")
			})
			
			sns.scatterplot(
				data = data,
				x = 'x',
				y = 'px',
				hue = 'group',
				palette = {
					'Inliers': 'tab:red',
					'Outliers': 'tab:blue'
				},
				s = 10,
				alpha = 1.0
			)
			
			plt.plot(L[:, 0], L[:, 1], '-', linewidth = 1.0, color = "tab:red")
			
			plt.xlabel('x')
			plt.ylabel('p_x')
			plt.show()

		p = p[~inliers]

	def lines_intersecton(line1, line2):
		a1, b1 = line1['a'], line1['b']
		a2, b2 = line2['a'], line2['b']
		
		_x = (b2 - b1) / (a1 - a2)
		_y = a1 * _x + b1
		return np.array([_x, _y])
	
	vertices = np.array(list(map(lambda i: lines_intersecton(lines_properties[i[0]], lines_properties[i[1]]), [[0, 1], [0, 2], [1, 2]])))

	if verbose == 2:
		import matplotlib.pyplot as plt
		import seaborn as sns

		with sns.axes_style("darkgrid"):

			sns.scatterplot(
				x = separatrix['x_unstable'],
				y = separatrix['px_unstable'],
				s = 10,
				color = "tab:red",
				label = "Unstable particle",
			)
			
			sns.scatterplot(
				x = separatrix['x_stable'],
				y = separatrix['px_stable'],
				s = 10,
				color = "tab:blue",
				label = "Stable particle",
			)

			plt.xlabel("x")
			plt.ylabel("px")

			plt.show()

		with sns.axes_style("darkgrid"):

			x_range = np.linspace(min(separatrix['x']), max(separatrix['x']))
			
			for i in range(3):
				line = lines_properties[i]['a'] * x_range + lines_properties[i]['b']
				line_y_range = (line > min(separatrix['px'])) & (line < max(separatrix['px']))
				
				plt.plot(x_range[line_y_range], line[line_y_range], '-', color = "tab:orange")

			plt.show()

	return DataFrame(dict(x = vertices[:, 0], px = vertices[:, 1]))

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

def find_triangle_vertices(x: np.ndarray, y: np.ndarray):
	points = np.column_stack((x, y))
	n_points = len(points)

	# the longest side of the triangle
	dist_max = 0
	vertice1_index, vertice2_index = 0, 0
	for i in range(n_points):
		for j in range(i + 1, n_points):
			dist = np.linalg.norm(points[i] - points[j])
			if dist > dist_max:
				dist_max = dist
				vertice1_index, vertice2_index = i, j

	vertice1 = points[vertice1_index]
	vertice2 = points[vertice2_index]

	def point_line_distance(x, linepoint1, linepoint2):
		return np.abs(np.cross(linepoint2 - linepoint1, x - linepoint1)) / np.linalg.norm(linepoint2 - linepoint1)

	# the furthest point from a line between
	# vertice1 and vertice2
	dist_max = 0
	vertice3_index = None
	for i in range(n_points):
		if i == vertice1_index or i == vertice2_index:
			continue
		dist = point_line_distance(points[i], vertice1, vertice2)
		if dist > dist_max:
			dist_max = dist
			vertice3_index = i
		
	vertice3 = points[vertice3_index]
	
	return np.array([vertice1, vertice2, vertice3])