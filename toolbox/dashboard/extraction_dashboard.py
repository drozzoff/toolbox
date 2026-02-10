import dash
from flask_compress import Compress
import plotly.graph_objs as go
import threading
import socket
import json
import sys
import numpy as np
from numpy.typing import NDArray
from functools import wraps
from pathlib import Path

from toolbox.dashboard.callbacks import register_callbacks
from toolbox.dashboard.layout import make_layout


def flatten_input(method):
	@wraps(method)
	def wrapper(self, values, *args, **kwargs):
		values = self._flatten(values)
		return method(self, values, *args, **kwargs)
	return wrapper

class DataBuffer:
	"""
	...
	"""
	def __init__(self):
		self.data, self.recent_data = [], []
		self.last_batch_id = -1

	def _flatten(self, values):
		# numpy scallar
		if isinstance(values, np.generic):
			return [values.item()]

		# numpy array
		if isinstance(values, np.ndarray):
			return values.tolist()
		
		# list
		if isinstance(values, list):
			return values

		# scallar
		return [values]

	@flatten_input
	def extend(self, values: list | NDArray, *, batch_id: int):
		self.recent_data = values
		self.data.extend(values)
		self.last_batch_id = batch_id

	@flatten_input
	def append(self, value, *, batch_id: int):
		self.recent_data = value
		self.data.append(value[0])
		self.last_batch_id = batch_id

	def clear(self):
		self.data.clear()
		self.recent_data = []
		self.last_batch_id = -1

	def last(self):
		return self.data[-1]

	def first(self):
		return self.data[0]

	def __str__(self):
		res = f"data = {self.data}\n"
		res += f"Received data since last update = {self.last_batch_id == -1}\n"
		if self.last_batch_id >= 0:
			res += f"New data received = {self.recent_data}\n"
		return res

	__repr__ = __str__

class ExtractionDashboard:
	"""
	Class to manage the tracking dashboard.
	"""

	CHUNK_SIZE = 1000 # max number points to send
	MAX_CALLBACK_LEVEL = 3
	
	def __init__(
			self,
			profile,
			host: str = '127.0.0.1',
			port: int = 0,
			data_to_monitor: list[str] | str | None = None,
		):
		self.profile = profile

		self.assets_dir = Path(__file__).resolve().parent / "assets"

		if host is None:
			raise ValueError("Host cannot be `None`.")
		
		if data_to_monitor is None:
			raise ValueError("No data to monitor provided")
		
		self.host, self.port = host, port
		self.data_to_monitor = data_to_monitor

		self._buflock = threading.Lock()

		if isinstance(data_to_monitor, str):
			self.data_to_monitor = [self.data_to_monitor]

		self._set_dependencies()

	def _set_dependencies(self):

		self.data_fields = self.profile.make_datafields(self)
		self.info_fields = self.profile.make_infofields(self)

		self.callbacks, buffer_keys = [], []
		print(self.data_to_monitor)

		self.info_dict = {}

		fields = self.data_fields | self.info_fields
		for data_key in self.data_to_monitor:
			if data_key not in fields:
				raise ValueError(f"Unsupported data requested: {data_key}. Supported data: {fields.keys()}")

			# Creating primary and secondary data buffers
			for key in fields[data_key].buffer_dependance:
				if not key in buffer_keys:
					buffer_keys.append(key)
			
			# creating output buffers for the callbacks
			if hasattr(fields[data_key], 'output_buffers') and fields[data_key].output_buffers is not None:
				for key in fields[data_key].output_buffers:
					if not key in buffer_keys:
						buffer_keys.append(key)

			# activating the field
			for field in [self.data_fields, self.info_fields]:
				try:
					field[data_key].state = True
				except KeyError: pass
			
			# saving the callbacks
			if fields[data_key].callback is not None:
				self.callbacks.append({
					'name': data_key,
					'level': fields[data_key].callback_level,
					'callback': fields[data_key].callback
				})
			
			if data_key in self.info_fields:
				tmp_dct = {x: None for x in self.info_fields[data_key].output_info}
				self.info_dict = self.info_dict | tmp_dct
		
		print(self.info_dict)

		self.callbacks.sort(key = lambda c: c['level'])

		# evaluating the data to be provided
		# and the buffers to be created
		# self.data_buffer contains the keys mixed of what user provides
		# and what is automatically generated in a callback
		fields_keys = list(fields.keys())

		buffers_to_create = buffer_keys.copy()

		buffer_keys_masked = list(filter(lambda key: key in fields_keys, buffer_keys))
		print(f"Pass 0: {buffer_keys}")
		print(f"\t not unique values = {buffer_keys_masked}")

		index = 1
		while buffer_keys_masked:
			for key in buffer_keys_masked:
				buffer_keys.remove(key)
				buffer_keys.extend(fields[key].buffer_dependance)
				buffers_to_create.extend(fields[key].buffer_dependance)

			buffer_keys = list(set(buffer_keys))

			print(f"Pass {index}: {buffer_keys}")
			index += 1
			buffer_keys_masked = list(filter(lambda key: key in fields_keys, buffer_keys))
			print(f"\t not unique values = {buffer_keys_masked}")

			if index == 10:
				raise Exception("Could not resolve dependencies.")
		
		self.data_to_expect = buffer_keys
		buffers_to_create = list(set(buffers_to_create))

		print(f"Data to expect = {self.data_to_expect}")
		print(f"buffers to create = {buffers_to_create}")
		
		self.data_buffer = {key: DataBuffer() for key in buffers_to_create}

	def _clear_buffer(self):
		# resetting the buffers in the memory
		with self._buflock:
			for key in self.data_buffer:
				self.data_buffer[key].clear()

		# resetting the pointers in the dependent data fields
		for data_key in self.data_fields:
			self.data_fields[data_key].buffer_pointer = 0
			self.data_fields[data_key].buffer_pointer_bin = 0

	def _buffers_filled_properly(self, batch_id: int):
		return all([self.data_buffer[key].last_batch_id == batch_id for key in self.data_to_expect])

	def run_callbacks(self):
		if not self._buffers_filled_properly(self.current_batch_id):
			raise ValueError(f"There is missing data for the batch {self.current_batch_id}")
		
		for i in range(self.MAX_CALLBACK_LEVEL):
			for callback in self.callbacks:
				if callback['level'] == i:
#					print(f"Running {callback['name']}")
					callback['callback']()

	def start_listener(self):
		srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self._listener_socket = srv
		try:
			srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
		except AttributeError:
			pass

		srv.bind((self.host, self.port))
		srv.listen(1)

		assigned_port = srv.getsockname()[1]
		print(f"[INFO] Started listener on port {assigned_port}")
		
		def run():
			while not getattr(self, "_stop_listener", False):
				conn, addr = srv.accept()
				print(f"[INFO] Connection from {addr}")
				self._clear_buffer()
				self.current_batch_id = 0

				buffer = ''
				try:
					while True:
						chunk = conn.recv(4096).decode()
						if not chunk:
							break
						buffer += chunk
						while '\n' in buffer:
							line, buffer = buffer.split('\n', 1)

							incoming = json.loads(line)

							with self._buflock:
								for key in self.data_to_expect:
									if key in incoming:
										self.data_buffer[key].extend(incoming[key], batch_id = self.current_batch_id)
								
								self.run_callbacks()
						
						self.current_batch_id += 1

				except json.JSONDecodeError as e:
					print("[ERROR] Invalid JSON", e)
				finally:
					conn.close()
					print("[INFO] Client disconnected, back to listening")
		
		self._listener_thread = threading.Thread(target = run, daemon = True)
		self._listener_thread.start()
	
	def stop_listener(self):
		if hasattr(self, '_stop_listener') and self._stop_listener:
			return

		self._stop_listener = True

		try:
			self._listener_socket.close()
		except Exception:
			pass	

	def plot_figure(self, key, **kwargs) -> go.Figure | None:
		fig = go.Figure()
		figure_config = self.data_fields[key].plot_order
		
		for i, tmp in enumerate(figure_config):
			x = kwargs.get(tmp['x'], [])
			y = kwargs.get(tmp['y'], [])

			if len(x) != len(y):
				print(f"[ERROR] length missmatch between x and y for '{key}' trace id = {i}")
				return

			fig.add_trace(go.Scatter(
				x = x,
				y = y,
				**tmp['settings']
			))

		self.data_fields[key].plot_layout(fig)

		return fig

	def run_dash_server(self):
		"""
		Start a dash server
		"""
		self.assets_dir = Path(__file__).resolve().parent / "assets"
		self.app = dash.Dash(__name__, title = "Extraction dashboard", assets_folder = str(self.assets_dir))

		Compress(self.app.server)

		self.app.layout = make_layout(self)

		register_callbacks(self.app, self)

		try:
			print("[INFO] Starting Dash server...")
			self.app.run(debug = True, use_reloader = False)
		except KeyboardInterrupt:
			print("\n[INFO] Caught Ctrl+C. Cleaning up...")
			sys.exit(0)

if __name__ == "__main__":
	from toolbox.dashboard.profiles import SIS18Profile

	test = ExtractionDashboard(
		profile = SIS18Profile(),
		port = 35235, 
		data_to_monitor = ["intensity"]
	)
	test.run_dash_server()