import numpy as np
import pickle as pk

def generate_bpsk(
	f0: float, 
	timestep: float, 
	signal_duration: float,
	chip_rate: float,
	**kwargs
	) -> tuple[np.array, np.array]:

	if "seed" in kwargs:
		np.random.seed(kwargs.get("seed"))

	timestamps = np.arange(0, signal_duration, timestep)
	carrier = np.sin(2 * np.pi * f0 * timestamps)
	signal = carrier

	n_samples = len(timestamps)

	n_chips = int(signal_duration * chip_rate)
	chip_values = np.random.choice([-1, 1], size = n_chips)

	chip_length = int(n_samples / n_chips)
	modulation = np.repeat(chip_values, chip_length)

	modulation = np.pad(modulation, (0, n_samples - len(modulation)), 'edge')

	signal = modulation * carrier

	if kwargs.get("save_for_debug", False):
		with open("data/ko_signal_recent.pk", 'wb') as f:
			pk.dump(dict(timestamps = timestamps, signal = signal), f)

	return timestamps, signal