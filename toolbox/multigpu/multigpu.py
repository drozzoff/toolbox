"""
Some simple functionality to distribute the tracking
over multiple GPUs that are available to the user.
"""

import warnings
from collections.abc import Callable
from pathlib import Path
import shutil
import multiprocessing as mp
import tempfile
import numpy as np
import pickle as pk
import xobjects as xo
import xtrack as xt
import time
import os

from toolbox.multigpu.worker import worker


def split_indices(n_elements: int, n_chuncks: int):
	base = n_elements // n_chuncks
	extra = n_elements % n_chuncks
	sizes = [base + (1 if i < extra else 0) for i in range(n_chuncks)]

	idx = np.cumsum([0] + sizes)
	return [(idx[i], idx[i+1]) for i in range(n_chuncks)]

def log_main(t0, msg, *, verbose: int = 1):
	if verbose < 1:
		return

	now = time.time() - t0
	print(f"[main] {now:.6f} s: {msg}", flush = True)

def track_multigpu(
	particles: xt.Particles | str,
	*, 
	line_constructor: Callable[[], xt.Line], 
	num_turns: int, 
	num_gpus: int,
	verbose: int = 1,
	record_every: int | None = None,
	checkpoint_every: int | None = None,
	monitor_budget: int | None = None,
	monitor_output_directory: str | Path | None = None,
	checkpoints_directory: str | Path | None = None
	) -> xt.Particles:
	"""
	Runs tracking on multiple GPUs. By default the workers use `xo.ContextPyopencl`
	context to do the tracking.

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
	verbose
	    Controls the output level.  
	    **0** - no output  
	    **1** - output from main process  
	    **2** - output from main and workers' processes
	record_every
	    If provided, uses `xt.ParticlesMonitor` to record the turn-by-turn data every `record_every`
	    given turn.
	checkpoint_every
	    If provided, every `checkpoint_every` turn save the `Particles` object to the specified directory
	monitor_budget
	    Maximum memory the monitor can occupy in memory when `record_every` is provided.
	monitor_output_directory
	    The directory to save the monitors data after the tracking. The worker saves 6d coordinates
	    along with `at_turn` and `state` for each particle. Creates a file for each GPU used in the
	    tracking
	checkpoints_directory
	    The folder to store the tracking checkpoints.
	Returns
	-------
	xt.Particles
	    The beam at the end of the tracking.
	"""
	if record_every is not None and record_every <= 0:
		raise ValueError("`record_every` must be positive")

	if record_every is not None and monitor_output_directory is None:
		raise ValueError("`monitor_output_directory` is required when recording")

	if checkpoint_every is not None and checkpoints_directory is None:
		raise ValueError("`checkpoints_directory` is required when saving checkpoints")

	if checkpoint_every is not None and checkpoint_every <= 0:
		raise ValueError(f"`checkpoint_every` must be positive, got {checkpoint_every}")

	if num_turns <= 0:
		raise ValueError(f"Number of turns should be positive, got {num_turns}")

	t0 = time.time()

	devices = xo.ContextPyopencl.get_devices()
	
	num_gpus_available = len(devices)
	if num_gpus_available < num_gpus:
		warnings.warn(
			f"Requested {num_gpus} GPUs but only {num_gpus_available} are available."
			f" Using {num_gpus_available} GPUs for the tracking"
		)
		num_gpus = num_gpus_available

	if num_gpus <= 0:
		raise ValueError(f"No GPUs requested/available.")

	devices = devices[:num_gpus]

	context = mp.get_context("spawn")
	
	tmp_folder = tempfile.TemporaryDirectory()
	temp_folder = tmp_folder.name

	if isinstance(particles, str):
		with open(particles, 'rb') as fid:
			particles = xt.Particles.from_dict(pk.load(fid))

	if particles._capacity < num_gpus:
		raise ValueError(
			f"Number of particles ({particles._capacity}) is smaller than number of"
			f" GPUs requested ({num_gpus})"
		)

	if checkpoint_every is not None:
		Path(checkpoints_directory).mkdir(parents = True, exist_ok = True)

	log_main(t0, "Start up", verbose = verbose)
	
	# splitting and saving the beam into chunks
	ranges = split_indices(particles._capacity, num_gpus)
	procs = []

	for device, (i0, i1) in zip(devices, ranges):
		mask = (particles.particle_id >= i0) & (particles.particle_id < i1)
		beam_chunk = particles.filter(mask)

		with open(os.path.join(temp_folder, f"in_beam_chunk_{device}.pkl") , 'wb') as fid:
			pk.dump(beam_chunk.to_dict(), fid)

	log_main(t0, "Saved the beam in the memory", verbose = verbose)

	for device, (i0, i1) in zip(devices, ranges):
		p = context.Process(
			target = worker,
			args = (
				line_constructor, 
				device, 
				num_turns, 
				temp_folder, 
				verbose, 
				record_every,
				checkpoint_every,
				monitor_budget,
				checkpoints_directory
			),
		)
		procs.append(p)

	log_main(t0, "Created workers' processes", verbose = verbose)

	for i, p in enumerate(procs):
		p.start()
		log_main(t0, f"Started process {i}", verbose = verbose)

	log_main(t0, f"Started all processes", verbose = verbose)

	__ = [p.join() for p in procs]
	
	log_main(t0, f"Processes joined", verbose = verbose)

	failed = [
		process.exitcode
		for process in procs
		if process.exitcode != 0
	]

	if failed:
		raise RuntimeError(f"Worker processes failed: {failed}")

	tracked_beam = None
	for device in devices:
		with open(os.path.join(temp_folder, f"out_beam_chunk_{device}.pkl"), 'rb') as fid:
			beam_chunk = xt.Particles.from_dict(pk.load(fid))

		if tracked_beam is not None:
			tracked_beam = xt.Particles.merge([tracked_beam, beam_chunk])
		else:
			tracked_beam = beam_chunk
	
	log_main(t0, f"Processed particles data", verbose = verbose)
	
	if record_every is not None:
		output_directory = Path(monitor_output_directory)
		output_directory.mkdir(parents = True, exist_ok = True)

		for device in devices:
			source = Path(temp_folder) / f"particle_monitor_{device}.h5"
			destination = output_directory / f"particle_monitor_{str(device).replace('.', '_')}.h5"

			shutil.copy2(source, destination)

		log_main(t0, f"Copied monitor data", verbose = verbose)
	
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