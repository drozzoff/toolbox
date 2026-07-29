import numpy as np
import xtrack as xt
from pandas import DataFrame


def get_phase_portrait2d(monitor: xt.ParticlesMonitor, particles: xt.Particles, at_turn: int, plane: str = 'x') -> DataFrame:
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

def get_phase_portrait4d(monitor: xt.ParticlesMonitor, particles: xt.Particles, at_turn: int) -> DataFrame:
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

def get_phase_portrait6d(monitor: xt.ParticlesMonitor, particles: xt.Particles, at_turn: int) -> DataFrame:
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

def compute_simple_masks(particles: xt.Particles):
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