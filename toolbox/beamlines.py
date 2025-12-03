import xtrack as xt
import numpy as np
import scipy as sp

from toolbox import generate_bpsk


def exc_freq_chirp(
	ctx,
	revolution_frequency: float,
	center_tune: float,
	tune_bandwidth: float,
	timestep: float, # or sampling rate
	excitation_duration: float | None,
	k0l: float,
	start_turn: int,
	**kwargs
	) -> xt.Exciter:

	excitation_frequency = revolution_frequency * center_tune
	width = revolution_frequency * tune_bandwidth

	time_stamps = np.arange(0, excitation_duration, timestep)

	samples = sp.signal.chirp(time_stamps, excitation_frequency - width, excitation_duration, excitation_frequency + width)

	return xt.Exciter(
		_context = ctx,
		samples = samples,
		sampling_frequency = 1 / timestep,
		duration = excitation_duration,
		frev = revolution_frequency,
		start_turn = start_turn,
		knl = [k0l],
		ksl = [0.0]
	)

def exc_bpsk(
	ctx,
	revolution_frequency: float,
	center_tune: float,
	tune_bandwidth: float,
	timestep: float, # or sampling rate
	excitation_duration: float | None,
	k0l: float,
	start_turn: int,
	**kwargs
	) -> xt.Exciter:

	excitation_frequency = revolution_frequency * center_tune
	chip_rate = revolution_frequency * tune_bandwidth

	timestamps, signal = generate_bpsk(
		f0 = excitation_frequency,
		timestep = timestep,
		signal_duration = excitation_duration,
		chip_rate = chip_rate,
		**kwargs
	)

	return xt.Exciter(
		_context = ctx,
		samples = signal,
		sampling_frequency = 1 / timestep,
		duration = excitation_duration,
		frev = revolution_frequency,
		start_turn = start_turn,
		knl = [k0l],
		ksl = [0.0]
	)

def _remove_inactive_multipoles_fix(line: xt.Line):
	"""
	Function to replace inactive thick mutipoles.
	Is needed because `Line.optimize_for_tracking()` does not handle them well.
	"""

	for ele, ele_name in zip(line, line.element_names):
		if isinstance(ele, xt.Multipole):
			aux = ([ele.hxl] + list(ele.knl) + list(ele.ksl))
			if np.sum(np.abs(np.array(aux))) == 0.0:
				if ele.isthick and ele.length != 0:
					line.remove(ele_name)

				
			
