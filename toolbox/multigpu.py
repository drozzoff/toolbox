"""
Some simple functionality to distribute the tracking
over multiple GPUs that are available to the user.
"""

import warnings
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import xobjects as xo
import xtrack as xt
import time


def split_indices(n_elements: int, n_chuncks: int):
	base = n_elements // n_chuncks
	extra = n_elements % n_chuncks
	sizes = [base + (1 if i < extra else 0) for i in range(n_chuncks)]

	idx = np.cumsum([0] + sizes)
	return [(idx[i], idx[i+1]) for i in range(n_chuncks)]

def create_shared_array(name: str, n: int, dtype = np.float64):
	itemsize = np.dtype(dtype).itemsize
	shm = shared_memory.SharedMemory(name = name, create = True, size = n * itemsize)
	arr = np.ndarray((n,), dtype = dtype, buffer = shm.buf)
	return shm, arr

def attach_shared_array(name: str, n: int, dtype = np.float64):
	shm = shared_memory.SharedMemory(name = name, create = False)
	arr = np.ndarray((n,), dtype = dtype, buffer = shm.buf)
	return shm, arr

def log_worker(t0, device, msg, *, verbose = True):
	if not verbose:
		return

	now = time.time() - t0
	print(f"[device = {device}] {now:.6f} s: {msg}", flush = True)

def worker(
	build_line: callable,
	device: str,
	num_turns: int,
	num_particles: int,
	i0: int,
	i1: int,
	shm_info: dict,
	out_shm_info: dict,
	verbose: bool = False,
	progress_queue = None,
	**kwargs
	):
	t0 = time.time()

	current_context =  xo.ContextPyopencl(device)
	log_worker(t0, device, "Created context", verbose = verbose)
	
	line_to_track = build_line()
	log_worker(t0, device, "Created a line", verbose = verbose)

	line_to_track.build_tracker(_context = current_context)
	line_to_track.optimize_for_tracking()
	
	log_worker(t0, device, "Built tracker", verbose = verbose)

	coords_to_track = {}

	for coord in shm_info:
		__, full_array = attach_shared_array(shm_info[coord]['in_name'], num_particles)
		coords_to_track[coord] = full_array[i0:i1].copy()

	log_worker(t0, device, "Prepared particles' coordinates", verbose = verbose)

	part_build_params = {}
	for key in {'nemitt_x', 'nemitt_y', 'method'}:
		if key in kwargs:
			part_build_params[key] = kwargs.get(key)
#	print(part_build_params)
	particles_to_track = line_to_track.build_particles(_context = current_context, **(part_build_params | coords_to_track))

	log_worker(t0, device, "Created particles", verbose = verbose)


	tmp = time.time()
	line_to_track.track(
		particles_to_track, 
		num_turns = num_turns,
		with_progress = True
	)

	log_worker(t0, device, f"Finished tracking | Track time = {time.time() - tmp}", verbose = verbose)

	for coord in out_shm_info:
		out_name = out_shm_info[coord]['out_name']
		__, out_array = attach_shared_array(out_name, num_particles)
		
		dev_arr = getattr(particles_to_track, coord)

		tracked_values = particles_to_track._context.nparray_from_context_array(dev_arr)

#		print(tracked_values, tracked_values.shape, type(tracked_values))
		out_array[i0:i1] = tracked_values

	log_worker(t0, device, "Extracted data to the shared memory", verbose = verbose)

def log_main(t0, msg, *, verbose = True):
	if not verbose:
		return

	now = time.time() - t0
	print(f"[main] {now:.6f} s: {msg}", flush = True)

def _clean_shm(shm_info: dict):
	for coord in shm_info:	
		for prefix in {'in', 'out'}:
			try:
				shm_name = shm_info[coord][f"{prefix}_name"]
				shm = shared_memory.SharedMemory(name = shm_name, create = False)
				shm.close()
				shm.unlink()
				print(f"--Cleaned {shm_name}")

			except (FileNotFoundError, KeyError):
				pass

def track_multigpu(
	*, 
	line_constructor: callable, 
	num_turns: int, 
	num_gpus: int, 
	with_progress = False, 
	verbose: bool = True,
	verbose_worker: bool = False,
	**kwargs):
	"""
	Runs tracking on GSI HPC with multiple GPUs.

	Similarly to `xtrack.track()` the coordinates passed must have the same length. But
	1-element arrays are acceptable, as it is expanded automatically.

	Parameters
	----------
	line_constructor
		A function that constructs `xt.Line` object that is going to be used to do the tracking.
		It needs to be a function since the workers will expand the latice on each context separately.
	num_turns
		Number of turns to do the tracking for.
	num_gpus
		Number of GPUs to use for the tracking.
	with_prgress
		If `True` a collective progress bar appears to sum up `.track()` from each context. *not implemented*
	verbose
		If `True` prints the timestamps of the program execution
	verbose_worker
		If `True` prints the timestamps of execution of each worker

	Additional parameters
	---------------------
	x : np.array
	px : np.array
	y : np.array
	py : np.array
	zeta : np.array
	delta : np.array
	x_norm : np.array
	px_norm : np.array
	y_norm : np.array
	py_norm : np.array
	zeta_norm : np.array
	delta_norm : np.array
	nemitt_x: float
	nemitt_y: float
	method: str

	"""
	t0 = time.time()

	mp.set_start_method("spawn", force = True)

	devices = xo.ContextPyopencl.get_devices()
	
	num_gpus_available = len(devices)
	if num_gpus_available < num_gpus:
		warnings.warn(f"Requested {num_gpus} GPUs but only {num_gpus_available} are available. Using {num_gpus_available} GPUs for the tracking")
		num_gpus = num_gpus_available

	devices = devices[:num_gpus]

	# accepted list of coordinates
	_coordinates_list = ['x', 'px', 'y', 'py', 'zeta', 'delta']
	_normalized_coordinates_list = ['x_norm', 'px_norm', 'y_norm', 'py_norm', 'zeta_norm', 'delta_norm'] # only for the input
	_accepted_coordinates_list = _coordinates_list + _normalized_coordinates_list

	log_main(t0, "Start up", verbose = verbose)

	# prebuild shm mapping
	shm_info = {}
	number_of_particles = 1
	for coord in _accepted_coordinates_list:
		if not coord in kwargs:
			continue

		n_elements = len(kwargs.get(coord))

		if number_of_particles == 1:
			number_of_particles = n_elements
		elif n_elements not in (1, number_of_particles):
			raise Exception("The arrays of the coordinates have incorrect size. The tracking will fail!")
		
		shm_info[coord] = {
			'in_name': f"{coord}_shm_in",
#			'out_name': f"{coord}_shm_out",
			'size': n_elements
		}

	# output coordinates are always fixed to _coordinates_list
	out_shm_info = {}

	for coord in _coordinates_list:
		out_shm_info[coord] = {
			'out_name': f"{coord}_shm_out",
			'size': n_elements
		}

	log_main(t0, "Calculated a shared memory mapping", verbose = verbose)

	# making sure, shared memo is not occupied
	_clean_shm(shm_info)
	_clean_shm(out_shm_info)

	log_main(t0, "Startup cleanup", verbose = verbose)

	for coord in shm_info:
		data = kwargs.get(coord)

		__, arr = create_shared_array(shm_info[coord]['in_name'], number_of_particles, np.float64)
		if shm_info[coord]['size'] == 1:
			arr[:] = np.full(number_of_particles, data[0], dtype = np.float64)
		else:
			arr[:] = data

	for coord in out_shm_info:
		__, out_arr = create_shared_array(out_shm_info[coord]['out_name'], number_of_particles, np.float64)
	
	log_main(t0, "Set partcicles' data in a shared memory", verbose = verbose)

	progress_queue = mp.Queue() if with_progress else None

	ranges = split_indices(number_of_particles, num_gpus)
	procs = []
	
	worker_kwargs = {}
	for key in {'nemitt_x', 'nemitt_y', 'method'}:
		if key in kwargs:
			worker_kwargs[key] = kwargs.get(key)
	for device, (i0, i1) in zip(devices, ranges):
		p = mp.Process(
			target = worker,
			args = (line_constructor, device, num_turns, number_of_particles, i0, i1, shm_info, out_shm_info, verbose_worker, progress_queue),
			kwargs = worker_kwargs
		)
		procs.append(p)
	
	log_main(t0, "Created workers' processes", verbose = verbose)

	for i, p in enumerate(procs):
		p.start()
		log_main(t0, f"Started process {i}", verbose = verbose)

	log_main(t0, f"Started all processes", verbose = verbose)

	__ = [p.join() for p in procs]
	
	log_main(t0, f"Processes joined", verbose = verbose)
	
	results = {}
	for coord in _coordinates_list:
		if coord not in shm_info:
			continue
		info = shm_info[coord]
		__, out_arr = attach_shared_array(info['out_name'], number_of_particles)
		results[coord] = np.array(out_arr) 

	log_main(t0, f"Finished", verbose = verbose)

	_clean_shm(shm_info)
	_clean_shm(out_shm_info)

	log_main(t0, f"Exit cleanup", verbose = verbose)

def line_constructor() -> xt.Line:

	reference_line = xt.Line(
		elements = [
			xt.Drift(length = 2.), 
			xt.Multipole(knl = [0, 0.5], ksl = [0, 0]),
			xt.Drift(length = 1.),
			xt.Multipole(knl = [0, -0.5], ksl = [0, 0])],
		element_names = ['drift_0', 'quad_0', 'drift_1', 'quad_1']
		)

	reference_line.particle_ref = xt.Particles(
		p0c = 6500e9,
		q0 = 1, 
		mass0 = xt.PROTON_MASS_EV
	)

	return reference_line

if __name__ == "__main__":

	n_part = int(4e5)

	track_multigpu(
		line_constructor = line_constructor,
		num_gpus = 3,
		num_turns = 100000,
		with_progress = False,
#		x = np.random.uniform(-1e-3, 1e-3, n_part),
#		px = np.random.uniform(-1e-5, 1e-5, n_part),
		x_norm = np.random.uniform(-1e-3, 1e-3, n_part),
		px_norm = np.random.uniform(-1e-5, 1e-5, n_part),
#		y = np.random.uniform(-2e-3, 2e-3, n_part),
#		py = np.random.uniform(-3e-5, 3e-5, n_part),
#		zeta = np.random.uniform(-1e-2, 1e-2, n_part),
#		delta = np.random.uniform(-1e-4, 1e-4, n_part),
		method = '4d',
		nemitt_x = 1.6e-5,
		nemitt_y = 2e-6,
		verbose_worker = True,

	)

	print("Tracking done")