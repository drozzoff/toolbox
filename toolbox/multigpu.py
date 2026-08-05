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
import h5py


def split_indices(n_elements: int, n_chuncks: int):
	base = n_elements // n_chuncks
	extra = n_elements % n_chuncks
	sizes = [base + (1 if i < extra else 0) for i in range(n_chuncks)]

	idx = np.cumsum([0] + sizes)
	return [(idx[i], idx[i+1]) for i in range(n_chuncks)]

def log_worker(t0, device, msg, *, verbose: int = 2):
	if verbose < 2:
		return

	now = time.time() - t0
	print(f"[device = {device}] {now:.6f} s: {msg}", flush = True)

def save_monitor_portion(
	output: h5py.File,
	monitor: xt.ParticlesMonitor,
	context: xo.ContextPyopencl,
	*,
	portion_index: int,
	portion_start: int,
	record_every: int,
	num_snapshots: int
	):
	group = output.create_group(f"portions/{portion_index:06d}")

	turns = portion_start + np.arange(num_snapshots, dtype = np.int64) * record_every

	group.create_dataset("turns", data = turns)

	for field in ("x", "px", "y", "py", "zeta", "delta", "state"):
		context_array = getattr(monitor, field)

		host_array = context.nparray_from_context_array(context_array)[..., 0]

		host_array = host_array.reshape(num_snapshots, -1)

		group.create_dataset(
			field,
			data = host_array,
			compression = "lzf",
			shuffle = True
		)

	group.attrs["record_every"] = record_every
	group.attrs["start_at_turn"] = portion_start
	group.attrs["num_snapshots"] = num_snapshots
	group.attrs["particle_id_start"] = int(monitor.part_id_start)
	group.attrs["particle_id_end"] = int(monitor.part_id_end)

def worker(
	build_line: Callable[[], xt.Line],
	device: str,
	num_turns: int,
	folder: str,
	verbose: int = 0,
	record_every: int | None = None,
	monitor_budget: int | None = None
	):
	"""
	If `record_every` is provided, the worker uses `xt.ParticlesMonitor` to
	record the data. The worker splits the total number of turns into multiple chunks
	of turns, so the memory the monitor occupies remains below the **limit**. 

	The limit is either given as an argument or is set based on the rule of thumb:
	```python
		monitor_budget = min(
		int(0.20 * device.global_mem_size),
		int(0.80 * device.max_mem_alloc_size)
	)

	Parameters
	----------
	build_line
		Function that expands the line
	device
		Name of the GPU device for the context
	num_turns
		Number of turns to track the particles for
	folder
		Name of the folder to read/write the particles' data
	verbose
		If `varbose > 1` prints the output from the worker
	record_every
		If provided will record the coordinates with `xt.ParticlesMonitor`
	monitor_budget
		Memory budget in bytes for the memory the `xt.ParticlesMonitor` is
		allowed to occupy.
		
	"""
	t0 = time.time()

	default_track_kwargs = {
		"time": True,
		"turn_by_turn_monitor": False,
		"with_progress": verbose > 1
	}

	log_worker(t0, device, "Created tracking configureation", verbose = verbose)
	
	current_context =  xo.ContextPyopencl(device)
	log_worker(t0, device, "Created context", verbose = verbose)

	line_to_track = build_line()
	log_worker(t0, device, "Created a line", verbose = verbose)

	line_to_track.build_tracker(_context = current_context)
	line_to_track.optimize_for_tracking(verbose = False)

	log_worker(t0, device, "Built tracker", verbose = verbose)

	with open(os.path.join(folder, f"in_beam_chunk_{device}.pkl"), 'rb') as fid:
		beam_chunk = xt.Particles.from_dict(pk.load(fid), _context = current_context)

	log_worker(t0, device, "Created particle beam", verbose = verbose)

	if record_every:
		if monitor_budget is None:
			monitor_budget = min(
				int(0.20 * current_context.device.global_mem_size),
				int(0.80 * current_context.device.max_mem_alloc_size)
			)

		bytes_per_record = sum(
			field_type._size
			for field_type, _ in xt.Particles.per_particle_vars
		)

		bytes_per_snapshot = beam_chunk._capacity * bytes_per_record

		particle_id_range = beam_chunk.get_active_particle_id_range()

		snapshots_budget = monitor_budget // bytes_per_snapshot

		if snapshots_budget < 1:
			raise MemoryError(
				"A single particle snapshot does not fit inside"
				f"The monitor budget of {monitor_budget / 2**30:.2f} GiB"
			)

		turns_budget = snapshots_budget * record_every

		n_max_budgets = num_turns // turns_budget
		rest_turns = num_turns % turns_budget
		turns_split = [turns_budget] * n_max_budgets
		if rest_turns:
			turns_split.append(rest_turns)

		monitor_filename = os.path.join(folder, f"particle_monitor_{device}.h5")

		_start_turn = 0
		with h5py.File(monitor_filename, "w") as output:
			for portion_index, num_turns_portion in enumerate(turns_split):
				monitor_kwargs = dict(
					particle_id_range = particle_id_range,
					start_at_turn = _start_turn,
					stop_at_turn = _start_turn + 1,
					repetition_period = record_every,
					n_repetitions = (num_turns_portion + record_every - 1) // record_every,
					auto_to_numpy = False
				)

				log_worker(
					t0, 
					device, 
					f"Portion {portion_index} | Monitor Created | Arguments = {monitor_kwargs}", 
					verbose = verbose
				)

				monitor = xt.ParticlesMonitor(
					_context = current_context,
					**monitor_kwargs
					)
				default_track_kwargs["turn_by_turn_monitor"] = monitor

				line_to_track.track(
					particles = beam_chunk, 
					num_turns = num_turns_portion,
					**default_track_kwargs
				)
				log_worker(
					t0, 
					device, 
					f"Portion {portion_index} | Finished tracking | Track time = {line_to_track.time_last_track}", 
					verbose = verbose
				)

				save_monitor_portion(	
					output,
					monitor, 
					current_context,
					portion_index = portion_index,
					portion_start = _start_turn,
					record_every = record_every,
					num_snapshots = (num_turns_portion + record_every - 1) // record_every
				)
				log_worker(
					t0, 
					device, 
					f"Portion {portion_index} | Monitor data saved", 
					verbose = verbose
				)

				_start_turn += num_turns_portion

				default_track_kwargs["turn_by_turn_monitor"] = False
				del monitor
	else:
		# default tracking without memory balancing
		line_to_track.track(
			particles = beam_chunk, 
			num_turns = num_turns,
			**default_track_kwargs
		)

	log_worker(t0, device, f"Finished tracking | Track time = {line_to_track.time_last_track}", verbose = verbose)

	with open(os.path.join(folder, f"out_beam_chunk_{device}.pkl"), 'wb') as fid:
		pk.dump(beam_chunk.to_dict(), fid)
	
	log_worker(t0, device, "Saved the chunk of the beam", verbose = verbose)

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
	monitor_budget: int | None = None,
	monitor_output_directory: str | Path | None = None
	) -> xt.Particles:
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
	phase_space_sampler
		A constructor for `PhaseSpaceSampler` object for the phase space evolution recording.
	Returns
	-------
	xt.Particles or tuple[xt.Particles, PhaseSpaceSampler]
		If `phase_space_sampler` is `None`, returns the tracked particles.

		If a phase-space sampler constructor is provided, returns a tuple containing:
		- The tracked particles.
		- The merged phase-space sampler containing snapshots from all workers.
	"""
	if record_every is not None and record_every <= 0:
		raise ValueError("`record_every` must be positive")

	if record_every is not None and monitor_output_directory is None:
		raise ValueError("`monitor_output_directory` is required when recording")

	t0 = time.time()

	mp.set_start_method("spawn", force = True)

	devices = xo.ContextPyopencl.get_devices()
	
	num_gpus_available = len(devices)
	if num_gpus_available < num_gpus:
		warnings.warn(f"Requested {num_gpus} GPUs but only {num_gpus_available} are available. Using {num_gpus_available} GPUs for the tracking")
		num_gpus = num_gpus_available

	devices = devices[:num_gpus]
	
	tmp_folder = tempfile.TemporaryDirectory()
	temp_folder = tmp_folder.name

	log_main(t0, "Start up", verbose = verbose)

	if isinstance(particles, str):
		with open(particles, 'rb') as fid:
			particles = xt.Particles.from_dict(pk.load(fid))
	
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
		p = mp.Process(
			target = worker,
			args = (
				line_constructor, 
				device, 
				num_turns, 
				temp_folder, 
				verbose, 
				record_every,
				monitor_budget
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
			destination = output_directory / f"particle_monitor_{str(device).replace(".", "_")}"

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