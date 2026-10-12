from collections.abc import Callable
import time
import os
import xobjects as xo
import xtrack as xt
import numpy as np
import pickle as pk
import h5py


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
	"""
	Dumps `xt.ParticlesMonitor` data into an existing `"h5"` file. Saves the following
	particles info into HDF5 file:
	```python
	{
		"particle_id",
		"x",
		"px",
		"y",
		"py",
		"zeta",
		"delta",
		"at_turn",
		"at_element",
		"state"
	}
	```
	Since its possible the tracking on the gpu is split into multiple chunks 
	of turns to keep the memory below the limit, multiple monitors might be used 
	to dump the data for 1 gpu.

	Parameters
	----------
	output
		Output file. Must be already created one
	monitor
		Monitor to save
	context
		Context the monitor exists at
	portion_index
		Portion id
	portion_start
		Initial turn of the recorded data
	record_every
		Defines how frequently the monitor recorded the data
	num_snapshots
		Number of snapshots in the monitor memory
	"""
	group = output.create_group(f"portions/{portion_index:06d}")

	turns = portion_start + np.arange(num_snapshots, dtype = np.int64) * record_every

	group.create_dataset("turns", data = turns)

	_fields = (
		"particle_id", 
		"x", "px", "y", "py", "zeta", "delta", 
		"at_turn", "at_element", "state"
	)

	for field in _fields:
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
	checkpoint_every: int | None = None,
	monitor_budget: int | None = None,
	checkpoints_folder: str | None = None
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

	When `checkpoint_every` is specified, dumps the `Particles` object back to the user 
	specified fodler every `checkpoint_every` number of turns. Its purpose is to provide a
	backup option if the tracking fails for some reason.

	Parameters
	----------
	build_line
		Function that expands the line
	device
		Name of the GPU device for the context
	num_turns
		Number of turns to track the particles for
	folder
		Name of the folder to read/write the particles' data. Typically is temporarily
		created folder
	verbose
		If `varbose > 1` prints the output from the worker
	record_every
		If provided will record the coordinates with `xt.ParticlesMonitor`
	checkpoint_every
		If provided, saves the particles object from the worker every given number of
		turns into the user space. Could be used as backup option, when the tracking
		can be interrupted. 
		**`checkpoint_every` is expected to be a multiple of `record_every`**.
	monitor_budget
		Memory budget in bytes for the memory the `xt.ParticlesMonitor` is
		allowed to occupy.
	checkpoints_folder
		Folder on the user side to store the latest checkpoint of the tracked `Particles`
		object. Is required when `checkpoint_every` is specified.	
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

	# defines the length of tracking in terms of turns that satisfies both 
	# `record_every` and `checkpoint_every`
	turns_budget = None

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

		# number of snapshots that can be done for the specified budget 
		snapshots_budget = monitor_budget // bytes_per_snapshot

		if snapshots_budget < 1:
			raise MemoryError(
				"A single particle snapshot does not fit inside"
				f"The monitor budget of {monitor_budget / 2**30:.2f} GiB"
			)

		# number of turns that fit into the specified monitor budget
		turns_budget = snapshots_budget * record_every

	# refining the turns_budget when the `checkpoint_every` is specified
	if checkpoint_every:
		if turns_budget is None:
			turns_budget = checkpoint_every
		else:
			turns_budget = min(turns_budget, checkpoint_every)

	if turns_budget is None:
		# default tracking without memory balancing or checkpoints
		line_to_track.track(
			particles = beam_chunk, 
			num_turns = num_turns,
			**default_track_kwargs
		)
	elif record_every:
		# `record_every` is specified.
		# `checkpoint_every` is also included if specified.

		n_max_budgets = num_turns // turns_budget
		rest_turns = num_turns % turns_budget
		turns_split = [turns_budget] * n_max_budgets
		if rest_turns:
			turns_split.append(rest_turns)

		monitor_filename = os.path.join(folder, f"particle_monitor_{device}.h5")

		number_portions = len(turns_split)

		_start_turn = 0
		checkpoint_id = 0
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
					f"Portion {portion_index}/{number_portions} | Monitor Created | Arguments = {monitor_kwargs}", 
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

				if checkpoint_every and (_start_turn % checkpoint_every == 0):
					# transfering the particles object back to the user
					with open(os.path.join(checkpoints_folder, f"checkpoint_{device}_{checkpoint_id:06d}.pkl"), 'wb') as fid:
						pk.dump(beam_chunk.to_dict(), fid)

					log_worker(
						t0, 
						device, 
						f"Checkpoint {checkpoint_id} | Saved | Turn = {_start_turn}", 
						verbose = verbose
					)
					checkpoint_id += 1

				default_track_kwargs["turn_by_turn_monitor"] = False
				del monitor
	else:
		# `checkpoint_every` is specified
		
		n_max_budgets = num_turns // turns_budget
		rest_turns = num_turns % turns_budget
		turns_split = [turns_budget] * n_max_budgets
		if rest_turns:
			turns_split.append(rest_turns)

		_start_turn = 0
		checkpoint_id = 0
		for portion_index, num_turns_portion in enumerate(turns_split):

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

			_start_turn += num_turns_portion
			

			# transfering the particles object back to the user
			with open(os.path.join(checkpoints_folder, f"checkpoint_{device}_{checkpoint_id:06d}.pkl"), 'wb') as fid:
				pk.dump(beam_chunk.to_dict(), fid)

			log_worker(
				t0, 
				device, 
				f"Checkpoint {checkpoint_id} | Saved | Turn = {_start_turn}", 
				verbose = verbose
			)

			checkpoint_id += 1

	log_worker(t0, device, f"Finished tracking", verbose = verbose)

	with open(os.path.join(folder, f"out_beam_chunk_{device}.pkl"), 'wb') as fid:
		pk.dump(beam_chunk.to_dict(), fid)
	
	log_worker(t0, device, "Saved the chunk of the beam", verbose = verbose)