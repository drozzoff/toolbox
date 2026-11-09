import multiprocessing as mp
import xobjects as xo


_worker_line = None

def _initialize_worker(line_factory, line_factory_kwargs):
	global _worker_line

	_worker_line = line_factory(**line_factory_kwargs)
	_worker_line.build_tracker(_context = xo.ContextCpu())

def get_worker_line():
	if _worker_line is None:
		raise RuntimeError("This process does not have initialized line")

	return _worker_line

class LineWorkerPool:

	def __init__(
		self,
		line_factory, 
		*, 
		line_factory_kwargs = None,
		processes = None,
		start_method = "spawn"
		):
		
		self.line_factory = line_factory
		self.line_factory_kwargs = (
			{}
			if line_factory_kwargs is None
			else line_factory_kwargs
		)

		self.processes = processes
		self.start_method = start_method
		self._pool = None

	def __enter__(self):
		context = mp.get_context(self.start_method)

		self._pool = context.Pool(
			processes = self.processes,
			initializer = _initialize_worker,
			initargs = (
				self.line_factory,
				self.line_factory_kwargs
			)
		)
		return self

	def __exit__(
		self,
	    exc_type,
	    exc_value,
	    traceback
		):
		pool = self._pool
		self._pool = None

		if pool is None:
			return False

		if exc_type is None:
			pool.close()
		else:
			pool.terminate()

		pool.join()

		return False

	def map(self, function, iterable, chunksize = None):
		if self._pool is None:
			raise RuntimeError("Worker pool is not running")

		return self._pool.map(
			function,
			iterable,
			chunksize
		)

	def apply(self, function, args = (), kwds = None):
		if self._pool is None:
			raise RuntimeError("worker pool is not running")

		return self._pool.apply(
			function,
			args = args,
			kwds = {} if kwds is None else kwds
		)