class NoStableParticleError(RuntimeError):
    """Expected failure"""

class NoUnstableParticleError(RuntimeError):
    """Expected failure"""

class SeparatrixSearchError(RuntimeError):
	"""Generic error when separatrix search fails for any reason"""

class SeptumCrossingError(RuntimeError):
	"""Error for when the septum crossing search fails"""