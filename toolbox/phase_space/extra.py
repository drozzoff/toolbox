import numpy as np
import xtrack as xt
from pandas import DataFrame


class PhaseSpaceSampler:
	"""
	The idea is that we reduce the resolution of the phase space. For instance to 64x64 or 100x100.
	It is not practically possible to keep the phase space information for any turn when number of particles
	used and number of turns are large. 
	**E.g.** Tracking on GPU of 250k particles for 1kk turns means to store
	**8 bytes x 2.5e4 x 1e6 ~ 2.0TB** of data in RAM. No computer would be able to do that.
	

	But phase space scalled down to a resolution of 100x100 in selected window and recorded every 1k turns means
	we only need **4 bytes  1000 x 100 x 100 ~ 40 MB** in `np.uint32` format.

	The way is to use it as a callback during the tracking.

	"""
	def __init__(self, 
		xlim: list,
		pxlim: list,
		every: int = 1_000,
		n_bins: int = 100,
		in_normalised_coordinates: bool = False,
		tw: xt.twiss.TwissTable | None = None,
		nemitt_x: float | None = None  
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
		in_normalised_coordinates
			If True, coordinates are assumed to be normalized coordinates. 
			If `True` `tw` must be provided too.
		tw
			If using normalized coordinates, Twiss table is used to convert the coodinates
		nemitt_x
			Normalised horizontal emittance, when normalised coordinates are used
		"""
		self.x_edges = np.linspace(*xlim, n_bins + 1)
		self.px_edges = np.linspace(*pxlim, n_bins + 1)
		self.in_normalised_coordinates = in_normalised_coordinates
		self.every = every
		self.tw = tw
		self.nemitt_x = nemitt_x
		
		self._call = 0
		self.turns = []
		self.histograms = []
		self.n_alive = []
		
	def __call__(self, line: xt.Line, particles:xt.Particles, **kwargs):
		"""
		The function used to do the logging inside `xtrack.Line.track()`.
		"""
		sample_now = (self._call % self.every) == 0
		turn = self._call
		self._call += 1

		if not sample_now:
			return None
		
		ctx = particles._context
		
		x = ctx.nparray_from_context_array(particles.x)
		px = ctx.nparray_from_context_array(particles.px)
		state = ctx.nparray_from_context_array(particles.state)
		alive = state > 0

		x_for_hist = x[alive]
		px_for_hist = px[alive]

		if self.in_normalised_coordinates:
			norm_coords = self.tw.get_normalized_coordinates(
				line.build_particles(x = x_for_hist, px = px_for_hist), 
				nemitt_x = self.nemitt_x, 
				nemitt_y = 0
			)
			x_for_hist = norm_coords.x_norm
			px_for_hist = norm_coords.px_norm
		
		hist, _, _ = np.histogram2d(
			x_for_hist,
			px_for_hist,
			bins = (self.x_edges, self.px_edges),
		)

		hist = hist.astype(np.uint32)
		alive_particles = np.sum(alive)
		
		self.turns.append(turn)
		self.histograms.append(hist)
		self.n_alive.append(alive_particles)

		return alive_particles