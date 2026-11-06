import warnings
import numpy as np
import xtrack as xt
from pandas import DataFrame
from skimage.measure import ransac, LineModelND
from toolbox.phase_space.separatrix_search import get_stable_and_unstable_particle, NoStableParticleError, NoUnstableParticleError

def sort_by_angle(x: np.array, px: np.array) -> tuple[np.array, np.array]:
	points = np.column_stack([x, px])
	
	center = points.mean(axis = 0)
	angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

	idx = np.argsort(angles)
	points_sorted = points[idx]

	return points_sorted[:, 0], points_sorted[:, 1]

def calcualte_enclosed_area(x: np.array, px: np.array) -> float:
	x_s, px_s = sort_by_angle(x, px)

	return 0.5 * np.abs(np.sum(x_s * np.roll(px_s, -1) - px_s * np.roll(x_s, -1)))

class SeptumCrossingError(RuntimeError):
	"""Error for when the septum crossing search fails"""

def calculate_septum_crossing(x: np.array, px: np.array, x_exit: float) -> dict:
	"""
	x_exit is the septum aperture
	"""
	alive_mask = (x != 0) & (px != 0)
	x_alive = x[alive_mask]
	px_alive = px[alive_mask]

	if x_alive.size < 2:
		raise SeptumCrossingError(
			"Cannot calculate the septum crossing. "
			f"The unstable trajectory contains {x_alive.size} points. "
			"At least 2 are needed."
		)

	x_mean = np.mean(x_alive)

	if x_exit <= x_mean:
		# septum is to the left of the particles
		extracted_id = np.argmin(x_alive)

	else:
		# septum is to the right of the particles
		extracted_id = np.argmax(x_alive)

	x0 = x_alive[extracted_id]
	px0 = px_alive[extracted_id]

	dist = np.sqrt((x - x0)**2 + (px - px0)**2)
	dist[extracted_id] = np.inf

	closest_id = np.argmin(dist)
	x1 = x_alive[closest_id]
	px1 = px_alive[closest_id]

	spiral_step = abs(x1 - x0)
	angle = px1 + (x_exit - x1) * (px0 - px1) / (x0 - x1)

	return {
		'spiral_step': spiral_step,
		'angle': angle,
		'last_point': [x0, px0],
		'prelast_point': [x1, px1],
	}

class SeparatrixSearchError(RuntimeError):
	"""Generic error when separatrix search fails"""

def get_separatrix_properties(
	line: xt.Line,
	septum: dict,
	ex_norm: float, 
	test_range: list[float] = [-2.0, 0.0],
	*,
	n_particles: int = 2, 
	num_turns: int = 500,
	ion: xt.Particles | None = None,
	precision: float = 1e-4,
	shrinkage_strength: int | str = "max",
	max_iterations: int = 2_000,
	twiss: xt.TwissTable | None = None,
	verbose: int = 1,
	) -> dict:
	"""
	Evaluates the separatrix enclosed area. If the septum location is provided
	also evaluates the spiral step and the extraction angle.
	
	Parameters
	----------
	line
		Beamline to do the tracking on
	ex_norm
		Horizontal normalized emittance
	septum
		Dict with the names of the entrance and exit apertures. Eg.:
		```python
		{
			"entrance": 'gs04me1e_en_aper',
			"exit": "gs04me1e_ex_aper"
		}
		```
	test_range
		initial guess for the location of the stable region limit. 
		Must contain two elements both negative. The number corresponds to the 
		normalized coordinate divided by a sigma. (`xt.line.build_particles()`))
	n_particles
		Number of particles used for the search. Default is 21.
	num_turns
		Number of turns to track for the search. Default is 1k.
	ion
		A particle to use for the tracking. 
		Default is `None` -> uses reference particle associated with a beamline.
	precision
		distance between the stable and unstable track. Same unit as `test_range`
	shrinkage_strength
		Defines how the `test_range` is shrinked in iterations.
		If `int` it leaves certain number of particles on the left and right 
		of the identified. If `str` is `'max'` no extra steps are added.
		Default is `'max'`.
	max_iterations
		Limit on the number of iterations for the stable area search and for RANSCAC.
		Default is `2000`.
	twiss : xt.TwissTable
		TwissTable for the lattice. If not provided, is evaluated internally.
	verbose : int
		If `verbose < 1` no output is printed.
		If `verbose == 1` prints the progress bar (**default**)
		If `verbose > 1` prints the progress bar and particles distribution 
			at each iteration

	Returns
	-------
	Dictionary containing enclosed area, extraction angle and a spiral step.
	Also includes bound and unbound trajectories in the phase space.		
	"""
	if twiss is None:
		twiss = line.twiss4d()

	try:
		particles, __ = get_stable_and_unstable_particle(
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
	except (NoStableParticleError, NoUnstableParticleError) as e:
		raise SeparatrixSearchError(
			f"`test_range` provided is bad."
			f"Caught {e} during the tracking."
			)
	except SeptumCrossingError:
		raise SeparatrixSearchError("Septum crossing evaluation failed")

	stable_particle, unstable_particle = particles

	p = stable_particle.copy()
	line.track(
		p,
		num_turns = num_turns,
		turn_by_turn_monitor = True
	)
	stable_rec = line.record_last_track
	x_stable = stable_rec.x.ravel()
	px_stable = stable_rec.px.ravel()

	area = calcualte_enclosed_area(x_stable, px_stable)

	p = unstable_particle.copy()
	line.track(
		p,
		num_turns = num_turns,
		turn_by_turn_monitor = True
	)
	unstable_rec = line.record_last_track
	x_unstable = unstable_rec.x.ravel()
	px_unstable = unstable_rec.px.ravel()

	if p.state[0] > 0:
		raise ValueError(f"Selected unstable particle survived all {num_turns} turns")

	# looking closely at the loss location
	ele_id_lost = p.at_element[0]
	ele_name_lost = line.element_names[ele_id_lost]

	septum_aperture = None

	if ele_name_lost in (septum['entrance'], septum['exit']):
		if ele_name_lost == septum['exit']:
			warnings.warn(
				"The unstable particle is lost on the exit aperture."
				"It is possible it is lost on the septum wires."
			)
		min_x = line[septum['entrance']].min_x
		max_x = line[septum['entrance']].max_x
		aperture_inf = 1e8

		if max_x > aperture_inf and abs(min_x) < aperture_inf:
			septum_aperture = min_x
		elif max_x < aperture_inf and abs(min_x) > aperture_inf:
			septum_aperture = max_x
		else:
			raise ValueError(
				f"Provided septum does not have one sided aperture"
			)
		
		sept_crossing = calculate_septum_crossing(x_unstable, px_unstable, septum_aperture)
	else:
		raise SeparatrixSearchError("Unstable particle is not extracted but lost in the line")

	if verbose == 2:
		import matplotlib.pyplot as plt
		import seaborn as sns

		x0, px0 = sept_crossing['last_point']
		x1, px1 = sept_crossing['prelast_point']

		with sns.axes_style("whitegrid"):		
			sns.scatterplot(
				x = x_unstable, y = px_unstable,
				s = 10, color = "tab:red",
				label = "Unstable track",
			)

			sns.scatterplot(
				x = x_stable, y = px_stable,
				s = 10, color = "black",
				label = "Stable track",
			)

			plt.axvline(
				x = septum_aperture, color = "tab:blue", 
				label = "Septum", lw = 2.0, 
				ls = 'dashed'
			)

			plt.plot(
				[x0, x1], [px0, px1], '-', 
			    color = "tab:orange", label = "Septum crossing")

			plt.xlabel("x [m]")
			plt.ylabel("px [rad]")
			plt.legend()
			
			plt.show()

	return {
		'area': area,
		'spiral_step': sept_crossing['spiral_step'],
		'angle': sept_crossing['angle']
	}

class ActionSeparatrix(xt.Action):
	"""
	An Action that can be used with default `xt.Line.match()` to match the 
	separatrix properties such as area

	"""
	def __init__(self, 
		line: xt.Line,
		septum: dict,
		ex_norm: float, 
		test_range: list[float] = [-2.0, 0.0],
		*,
		n_particles: int = 2, 
		num_turns: int = 500,
		ion: xt.Particles | None = None,
		precision: float = 1e-4,
		shrinkage_strength: int | str = "max",
		max_iterations: int = 2_000,
		twiss: xt.TwissTable | None = None,
		verbose: int = 0,
		):

		for name, value in locals().copy().items():
			if name != "self":
				setattr(self, name, value)

	def run(self) -> dict:
		try:
			res = get_separatrix_properties(**self.__dict__)
		except SeparatrixSearchError:
			return 'failed'

		return res

def get_separatrix_vertices(
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
	residual_threshold: float = 1e-4,
	**kwargs
	) -> DataFrame:
	"""
	Calculates 3 vertices of the separatrix around 3rd integer resonance.
	Uses the RANSAC (random sample consensus) algorithm for the fit.
	
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
		A particle to use for the tracking. 
		Default is `None` -> uses reference particle associated with a beamline.
	precision
		distance between the stable and unstable track. `test_range` is used to 
		evaluate this.
	shrinkage_strength
		Defines how the `test_range` is shrinked in iterations.
		If `int` it leaves certain number of particles on the left and right 
		of the identified. If `str` is `'max'` no extra steps are added.
		Default is `'max'`.
	max_iterations
		Limit on the number of iterations for the stable area search and for RANSCAC.
		Default is `2000`.
	residual_threshold:
		Maximum distance for a data point to be classified as an inlier.
		**passed to RANSAC**.
		
	Additional parameters
	---------------------
	verbose : int
		If `verbose < 1` no output is printed.
		If `verbose == 1` prints the progress bar (**default**)
		If `verbose > 1` prints the progress bar and particles distribution 
			at each iteration
	zeta : np.array
		`zeta` coordinate for the test beam
	delta : np.array
		`delta` coordinate for the test beam

	Returns
	-------
	DataFrame
		Coordinates of 3 points in the phase space being the vertices of the 
		stability triangle.
	"""
	verbose = kwargs.get('verbose', 0)

	particles, __ = get_stable_and_unstable_particle(
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
	stable_particle, unstable_particle = particles

	p = stable_particle.copy()
	line.track(p, num_turns = num_turns, turn_by_turn_monitor = True)
	stable_rec = line.record_last_track

	p = unstable_particle.copy()
	line.track(p, num_turns = num_turns, turn_by_turn_monitor = True)
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

	for i in range(3):
		model, inliers = ransac(
			p, 
			LineModelND, 
			min_samples = 2, 
			residual_threshold = residual_threshold,
			max_trials = max_iterations
		)

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
	
	vertices = np.array(list(map(
		lambda i: lines_intersecton(lines_properties[i[0]], lines_properties[i[1]]), 
		[[0, 1], [0, 2], [1, 2]]
	)))

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
