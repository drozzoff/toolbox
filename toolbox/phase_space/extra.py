import numpy as np
import h5py
from pathlib import Path
import tqdm

class PhaseSpaceSnapshot:
	def __init__(self, 
		x_edges: np.ndarray,
		px_edges = np.ndarray,
		*,
		at_turn: int | None = None,
		x: np.ndarray | None = None, 
		px: np.ndarray | None = None, 
		state: np.ndarray | None = None,
		):
		"""
		Parameters
		----------
		xlim
			2 elements array with sampling range for `x`
		pxlim
			2 elements array with sampling range for `px`
		every
			Frequency of the phase space snapshots
		n_bins
			Number of bins to use in each plane
		"""
		self.x_edges = x_edges
		self.px_edges = px_edges

		self.at_turn = at_turn
		self.histogram = None
		self.n_alive = None

		if self.at_turn is not None:
			self.calculate_histogram(x, px, state)
		
	def calculate_histogram(
		self,
		x: np.ndarray, 
		px: np.ndarray, 
		state: np.ndarray,
		):
		"""
		"""
		alive = state > 0

		x_for_hist = x[alive]
		px_for_hist = px[alive]

		hist, _, _ = np.histogram2d(
			x_for_hist,
			px_for_hist,
			bins = (self.x_edges, self.px_edges),
		)

		self.histogram = hist.astype(np.uint32)
		self.n_alive = np.count_nonzero(alive)

class PhaseSpaceSnapshots:
	"""
	The idea is that we reduce the resolution of the phase space. For instance to 64x64 or 100x100.
	It is not practically possible to keep the phase space information for any turn when number of particles
	used and number of turns are large. 
	**E.g.** Tracking on GPU of 250k particles for 1kk turns means to store
	**8 bytes x 2.5e4 x 1e6 ~ 2.0TB** of data in RAM.

	But phase space scalled down to a resolution of 100x100 in selected window and recorded every 1k turns means
	we only need **4 bytes  1000 x 100 x 100 ~ 40 MB** in `np.uint32` format.
	"""
	def __init__(self, 
		xlim: list,
		pxlim: list,
		n_bins: int = 100,
		*,
		filename: str | Path
		):

		self.x_edges = np.linspace(*xlim, n_bins + 1)
		self.px_edges = np.linspace(*pxlim, n_bins + 1)

		self.histograms = []
		self.turns = []
		self.n_alive = []

		if filename is not None:
			self.process_monitor_data(filename)

	def process_monitor_data(self, filename: str | Path):
		with h5py.File(filename, "r") as f:
			portions = f["portions"]

			for portion_name in tqdm(portions):
				group = portions[portion_name]

				turns = group["turns"]

				for i, at_turn in enumerate(turns):
					snapshot = PhaseSpaceSnapshot(
						self.x_edges,
						self.px_edges,
						at_turn = at_turn,
						x = group["x"][i],
						px = group["px"][i],
						state = group["state"][i],
					)

					self.histograms.append(snapshot.histogram)
					self.turns.append(snapshot.at_turn)
					self.n_alive.append(snapshot.n_alive)

	def __iadd__(self, other):
		if not isinstance(other, PhaseSpaceSnapshots):
			return NotImplemented

		np.testing.assert_array_equal(self.x_edges, other.x_edges)
		np.testing.assert_array_equal(self.px_edges, other.px_edges)
		np.testing.assert_array_equal(self.turns, other.turns)

		if len(self.histograms) != len(other.histograms):
			raise ValueError("Number of snapshots does not match.")

		self.histograms = [
			h1 + h2
			for h1, h2 in zip(self.histograms, other.histograms)
		]

		self.n_alive = [
			n1 + n2
			for n1, n2 in zip(self.n_alive, other.n_alive)
		]

		return self

	def __add__(self, other):
		if not isinstance(other, PhaseSpaceSnapshots):
			return NotImplemented

		np.testing.assert_array_equal(self.x_edges, other.x_edges)
		np.testing.assert_array_equal(self.px_edges, other.px_edges)
		np.testing.assert_array_equal(self.turns, other.turn)

		if len(self.histograms) != len(other.histograms):
			raise ValueError("Number of snapshots does not match.")

		result = PhaseSpaceSnapshots(
			xlim = [self.x_edges[0], self.x_edges[-1]],
			pxlim = [self.px_edges[0], self.px_edges[-1]],
			n_bins = len(self.x_edges) - 1,
			filename = None,
		)

		result.turns = list(self.turns)

		result.histograms = [
			h1 + h2
			for h1, h2 in zip(self.histograms, other.histograms)
		]

		result.n_alive = [
			n1 + n2
			for n1, n2 in zip(self.n_alive, other.n_alive)
		]

		return result
					
	def save_data(self, filename: str | Path = "phase_space_snapshots.h5"):
		"""
		Save the data in HDF5 format in the following structure
		phase_space_snapshots.h5
		├── histograms		shape: (n_snapshots, n_x_bins, n_px_bins), dtype uint32
		├── turns			shape: (n_snapshots,), dtype int64
		├── n_alive			shape: (n_snapshots,), dtype uint32
		├── x_edges			shape: (n_x_bins + 1,)
		├── px_edges		shape: (n_px_bins + 1,)
		└── attrs
			└── 
		"""
		with h5py.File(filename, "w") as f:
			f.create_dataset(
				"histograms",
				data = np.asarray(self.histograms, dtype = np.uint32),
				compression = "gzip",
				shuffle = True
			)
			f.create_dataset("turns", data = np.asarray(self.turns, dtype = np.int64))
			f.create_dataset("n_alive", data = np.asarray(self.n_alive, dtype = np.uint32))
			f.create_dataset("x_edges", data = self.x_edges)
			f.create_dataset("px_edges", data = self.px_edges)