from functools import partial
import numpy as np
from numpy.typing import NDArray


def generate_bpsk(
	f0: float, 
	timestamps: NDArray[np.floating],
	chip_rate: float,
	**kwargs
	) -> NDArray[np.float32]:
	
	if "seed" in kwargs:
		np.random.default_rng(kwargs.get("seed"))
	
	timestep = timestamps[1] - timestamps[0]
	duration = timestamps[-1] - timestamps[0]
	
	carrier = np.sin(2 * np.pi * f0 * timestamps)
	signal = carrier
	
	n_samples = len(timestamps)
	n_chips = int(duration * chip_rate) + 1

	if n_chips == 1:
		raise ValueError(f"Number of chips is too low! Increase the signal duration or the chirp_rate.")
	
	chip_values = np.random.choice([-1, 1], size = n_chips)
	chip_length = int(1 / (timestep * chip_rate))
	modulation = np.repeat(chip_values, chip_length)[:n_samples]

	modulation = np.pad(modulation, (0, n_samples - len(modulation)), 'edge')
	
	signal = modulation * carrier
	
	signal = signal.astype(np.float32)
	return signal

def u_pp_model(A_start, A_end, tau_start, tau_end, t):
	alpha = 61.081231

	u1 = (A_start - A_end / np.exp(1)) * np.exp(-t / (4 * tau_start))
	u2 = A_end * np.exp(t / (2 * tau_end) - 1)
	return alpha * (u1 + u2)

def u_pp_windowed(t, *, A_start, A_end, tau_start, tau_end, spill_time):
	ramp_time = 0.032
	
	t = np.asarray(t, dtype = float)
	res = np.zeros_like(t, dtype = float)
	
	eval_func = partial(u_pp_model, A_start, A_end, tau_start, tau_end)
	
	ramp_up_mask = t <= ramp_time
	ramp_down_mask = (t >= (spill_time - ramp_time)) & (t <= spill_time)
	inside_ramp = (t > ramp_time) & (t < spill_time - ramp_time)

	ramp_up_peak = eval_func(ramp_time)
	ramp_down_peak = eval_func(spill_time - ramp_time)
	
	res[ramp_up_mask] = ramp_up_peak * t[ramp_up_mask] / ramp_time
	res[inside_ramp] = eval_func(t[inside_ramp])
	res[ramp_down_mask] = ramp_down_peak * (spill_time - t[ramp_down_mask]) / ramp_time

	return res.item() if np.isscalar(t) else res

def plain_bpsk(
	*,
	frev: float,
	Qx: float,
	Qx_bandwidth: float,
	timestamps: NDArray[np.floating],
	filename: str | None = None,
	verbose: int = 0
	) -> NDArray[np.float32]:
	"""
	Plain RBPSK signal, with normalized amplitude.

	Parameters
	----------
	frev
		Beam revolution frequency
	Qx
		Horizontal tune of the beam
	Qx_bandwidth
		Horizontal tune bandwidth to be covered by the generated BPSK
	timestamps
		Timestamps for the signal
	filename
		If provided, the location to save the signal
	verbose
		If other than 0 (default = 0) prints some info

	Returns
	-------
	NDArray[np.float32]
		Created BPSK.
	"""

	excitation_frequency = frev * Qx
	chip_rate = frev * Qx_bandwidth

	timestep = timestamps[1] - timestamps[0]
	duration = timestamps[-1] - timestamps[0]
	
	if verbose:
		print(f"RBPSK properties:\n\tCarrier = {(excitation_frequency * 1e-6):.4f} MHz; Chip rate = {(chip_rate * 1e-3):.4f} kHz")
		print(f"\tDuration = {(duration):.5f} s; Sampled at {(1 / timestep * 1e-6):.4f} MHz")
	
	signal = generate_bpsk(
		f0 = excitation_frequency,
		timestamps = timestamps,
		chip_rate = chip_rate
	)
	if filename:
		np.save(filename, signal)
		
	return signal

def modulated_bpsk(
	*,
	frev: float,
	Qx: float,
	Qx_bandwidth: float,
	A_start: float,
	A_end: float,
	tau_start: float,
	tau_end: float,
	spill_time: float,
	timestamps: NDArray[np.floating],
	filename: str | None = None,
	verbose: int = 0
	) -> tuple[float, NDArray[np.float32]]:
	"""
	Amplitude modulated RBPSK. The same way it is implemented in SIS18 control.

	Parameters
	----------
	frev
		Beam revolution frequency
	Qx
		Horizontal tune of the beam
	Qx_bandwidth
		Horizontal tune bandwidth to be covered by the generated BPSK
	A_start
		Amplitude of the decaying exponent
	A_end
		Amplitude of the rising exponent
	tau_start
		Decay time
	tau_end
		Rise time
	spill_time
		Excitation duration, including the fixed ramp time (2 * 0.032)
	timestamps
		Timestamps for the signal
	filename
		If provided, the location to save the signal
	verbose
		If other than 0 (default = 0) prints some info
	"""
	bpsk = plain_bpsk(
		frev = frev,
		Qx = Qx,
		Qx_bandwidth = 	Qx_bandwidth,
		timestamps = timestamps,
		verbose = verbose
	)
	
	amplitude_modulation_func = partial(
		u_pp_windowed,
		A_start = A_start,
		A_end = A_end,
		tau_start = tau_start,
		tau_end = tau_end,
		spill_time = spill_time
	)
	amplitude_modulation = amplitude_modulation_func(timestamps)

	U_pp_max = max(amplitude_modulation)
	amplitude_modulation /= U_pp_max

	modulated_signal = bpsk * amplitude_modulation
	
	if filename:
		np.save(filename, modulated_signal)

	return U_pp_max, modulated_signal
