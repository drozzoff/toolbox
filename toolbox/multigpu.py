"""
Some simple functionality to distribute the tracking
over multiple GPUs that are available to the user.
"""

import warnings
import multiprocessing as mp
import tempfile
import numpy as np
import pickle as pk
import xobjects as xo
import xtrack as xt
import time


def split_indices(n_elements: int, n_chuncks: int):
	base = n_elements // n_chuncks
	extra = n_elements % n_chuncks
	sizes = [base + (1 if i < extra else 0) for i in range(n_chuncks)]

	idx = np.cumsum([0] + sizes)
	return [(idx[i], idx[i+1]) for i in range(n_chuncks)]

def log_worker(t0, device, msg, *, verbose = True):
	if not verbose:
		return

	now = time.time() - t0
	print(f"[device = {device}] {now:.6f} s: {msg}", flush = True)

def worker(
	build_line: callable,
	particles_filename: str,
	device: str,
	num_turns: int,
	i0: int, # including
	i1: int, # excluding
	folder_to_save_particles: str,
	verbose: bool = False
	):
	t0 = time.time()

	current_context =  xo.ContextPyopencl(device)
	log_worker(t0, device, "Created context", verbose = verbose)
	
	line_to_track = build_line()
	log_worker(t0, device, "Created a line", verbose = verbose)

	line_to_track.build_tracker(_context = current_context)
	line_to_track.optimize_for_tracking(verbose = False)

	log_worker(t0, device, "Built tracker", verbose = verbose)

	with open(particles_filename, 'rb') as fid:
		main_beam = xt.Particles.from_dict(pk.load(fid), _context = current_context)
	mask = (main_beam.particle_id >= i0) & (main_beam.particle_id < i1)
	beam_chunk = main_beam.filter(mask)

	log_worker(t0, device, "Created particle beam", verbose = verbose)

	line_to_track.track(
		beam_chunk, 
		num_turns = num_turns,
		time = True,
		with_progress = verbose,
	)

	log_worker(t0, device, f"Finished tracking | Track time = {line_to_track.time_last_track}", verbose = verbose)

	with open(f"{folder_to_save_particles}/beam_chunk_{device}.pkl", 'wb') as fid:
		pk.dump(p.to_dict(), fid)
	
	log_worker(t0, device, "Saved the chunk of the beam", verbose = verbose)

def log_main(t0, msg, *, verbose = True):
	if not verbose:
		return

	now = time.time() - t0
	print(f"[main] {now:.6f} s: {msg}", flush = True)

def track_multigpu(
	particles: xt.Particles | str,
	*, 
	line_constructor: callable, 
	num_turns: int, 
	num_gpus: int,
	verbose: int = 1
	):
	"""
	Runs tracking on GSI HPC with multiple GPUs.

	Similarly to `xtrack.track()` the coordinates passed must have the same length. But
	1-element arrays are acceptable, as it is expanded automatically.

	Parameters
	----------
	particles
		`Particles` object to track
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
		Controls the output level.
		`0` - no output
		`1` - output from main process
		`2` - output from main and workers' processes

	Returns
	-------
	xt.Particles
		A new object `xt.Particles` with the data at the end of the tracking.
	"""
	t0 = time.time()

	mp.set_start_method("spawn", force = True)

	devices = xo.ContextPyopencl.get_devices()
	
	num_gpus_available = len(devices)
	if num_gpus_available < num_gpus:
		warnings.warn(f"Requested {num_gpus} GPUs but only {num_gpus_available} are available. Using {num_gpus_available} GPUs for the tracking")
		num_gpus = num_gpus_available

	devices = devices[:num_gpus]
	verbose_worker = verbose > 1

	log_main(t0, "Start up", verbose = verbose)
	if isinstance(particles, xt.Particles):
		temp_folder = tempfile.TemporaryDirectory()
		folder_to_save_particles = temp_folder.name
		main_beam_loc = f"{folder_to_save_particles}/main_beam.pkl"
		with open(main_beam_loc, 'wb') as fid:
			pk.dump(particles.to_dict(), fid)
	
	if isinstance(particles, str):
		main_beam_loc = particles

	log_main(t0, "Saved the beam in the memory", verbose = verbose)

	ranges = split_indices(particles._capacity, num_gpus)
	procs = []
	
	for device, (i0, i1) in zip(devices, ranges):
		p = mp.Process(
			target = worker,
			args = (line_constructor, main_beam_loc, device, num_turns, i0, i1, folder_to_save_particles, verbose_worker),
		)
		procs.append(p)

	log_main(t0, "Created workers' processes", verbose = verbose)

	for i, p in enumerate(procs):
		p.start()
		log_main(t0, f"Started process {i}", verbose = verbose)

	log_main(t0, f"Started all processes", verbose = verbose)

	__ = [p.join() for p in procs]
	
	log_main(t0, f"Processes joined", verbose = verbose)

	tracked_beam = None
	for device in devices:
		with open(f"{folder_to_save_particles}/beam_chunk_{device}.pkl", 'rb') as fid:
			beam_chunk = xt.Particles.from_dict(pk.load(fid))

		if tracked_beam:
			tracked_beam = xt.Particles.merge([tracked_beam, beam_chunk])
		else:
			tracked_beam = beam_chunk
	
	log_main(t0, f"Processed results", verbose = verbose)
	
	log_main(t0, f"Finished", verbose = verbose)

	return tracked_beam

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

	line = line_constructor()
	
	n_part = int(1e4)

	p = line.build_particles(
		x = np.random.uniform(-1e-3, 1e-3, n_part),
		px = np.random.uniform(-1e-5, 1e-5, n_part),
#		x_norm = np.random.uniform(-1e-3, 1e-3, n_part),
#		px_norm = np.random.uniform(-1e-5, 1e-5, n_part),
		y = np.random.uniform(-2e-3, 2e-3, n_part),
		py = 0.0,
#		zeta = np.random.uniform(-1e-2, 1e-2, n_part),
#		delta = np.random.uniform(-1e-4, 1e-4, n_part),
		method = '4d',
#		nemitt_x = 1.6e-5,
#		nemitt_y = 2e-6,
	)

	p_tracked = track_multigpu(
		p,
		line_constructor = line_constructor,
		num_gpus = 1,
		num_turns = int(1e4),
		verbose = 0
	)

	print(p.x, p.px)
	print(p_tracked.x, p_tracked.px)