import dash
from dash import dcc, html, no_update, MATCH
from dash.dependencies import Output, Input, State
from flask_compress import Compress
import plotly.graph_objs as go
import threading
import socket
import json
import sys
import os
import numpy as np
from numpy.typing import NDArray
from functools import wraps
from pathlib import Path
import traceback
import pandas as pd
import datetime
import numbers
from toolbox.dashboard.profiles.datafield import Ratio


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

	def __str__(self):
		res = f"data = {self.data}\n"
		res += f"Received data since last update = {self.last_batch_id == -1}\n"
		if self.last_batch_id >= 0:
			res += f"New data received = {self.recent_data}\n"
		return res

	__repr__ = __str__

def _bin_array(arr, bin_length: int, how: str) -> list:
	if not arr: return []
	
	bins_length = (len(arr) // bin_length) * bin_length
	if bins_length == 0: return []
	
	if how == "first":
		return arr[:bins_length:bin_length]
	if how == "last":
		return arr[bin_length - 1:bins_length:bin_length]
	if how == "middle":
		return arr[bin_length // 2:bins_length:bin_length]
	
	if isinstance(arr[0], numbers.Real):
		a = np.asarray(arr[:bins_length])
		a_bined = a.astype(float, copy = False).reshape(-1, bin_length)
		
		if how == "sum":
			return a_bined.sum(axis = 1).tolist()
		if how == "mean":
			return a_bined.mean(axis = 1).tolist()	
		if how == "max":
			return a_bined.max(axis = 1).tolist()
		if how == "min":
			return a_bined.min(axis = 1).tolist()
		
		raise ValueError(how)
	elif isinstance(arr[0], Ratio):
		print("THe data is Ratio")
		out = []
		for i in range(0, bins_length, bin_length):
			chunk = arr[i: i + bin_length]

			if how in {"sum", "mean"}:
				res = sum(chunk, Ratio(0, 0))
				out.append(res.value())
			elif how == "max":
				out.append(max(x.value() for x in chunk))
			elif how == "min":
				out.append(min(x.value() for x in chunk))
			else:
				raise ValueError(how)
		return out
	else:
		raise TypeError(f"Unsupported data type: {type(arr[0])}")

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
		self.callbacks = []

		print(self.data_to_monitor)

		buffer_keys = []

		for data_key in self.data_to_monitor:
			if data_key not in self.data_fields:
				raise ValueError(f"Unsupported data requested: {data_key}. Supported data: {self.data_fields.keys()}")

			# Creating primary and secondary data buffers
			for key in self.data_fields[data_key].buffer_dependance:
				if not key in buffer_keys:
					buffer_keys.append(key)
			
			# creating output buffers for the callbacks
			if self.data_fields[data_key].output_buffers is not None:
				for key in self.data_fields[data_key].output_buffers:
					if not key in buffer_keys:
						buffer_keys.append(key)

			# activating the DataField
			self.data_fields[data_key].state = True

			# saving the callbacks
			if self.data_fields[data_key].callback is not None:
				self.callbacks.append({
					'name': data_key,
					'level': self.data_fields[data_key].callback_level,
					'callback': self.data_fields[data_key].callback
				})
		self.callbacks.sort(key = lambda c: c['level'])

		# evaluating the data to be provided
		# and the buffers to be created
		# self.data_buffer contains the keys mixed of what user provides
		# and what is automatically generated in a callback
		datafields_keys = list(self.data_fields.keys())

		buffers_to_create = buffer_keys.copy()

		buffer_keys_masked = list(filter(lambda key: key in datafields_keys, buffer_keys))
		print(f"Pass 0: {buffer_keys}")
		print(f"\t not unique values = {buffer_keys_masked}")

		index = 1
		while buffer_keys_masked:
			for key in buffer_keys_masked:
				buffer_keys.remove(key)
				buffer_keys.extend(self.data_fields[key].buffer_dependance)
				buffers_to_create.extend(self.data_fields[key].buffer_dependance)

			buffer_keys = list(set(buffer_keys))

			print(f"Pass {index}: {buffer_keys}")
			index += 1
			buffer_keys_masked = list(filter(lambda key: key in datafields_keys, buffer_keys))
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

		intro_text = '''
		### SIS18 slow extraction dashboard.
		'''
		divs = {
			'turn_dependent_data': [],
			'phase_space': [],
			'separatrix': [],
			'special': []
		}
		for key in self.data_to_monitor:
			
			# Tab 1 - Turn dependent data
			if self.data_fields[key].category == "Turn By Turn":
				divs['turn_dependent_data'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = self.plot_figure(key, init_run = True),
							style = {"width": "100%"},
							config = {"responsive": True}
						)
					], style = {'display': 'flex', 'gap': '10px'})
				)
			
			# Tab 2 - Phase space
			if self.data_fields[key].category == "Phase Space":
				divs['phase_space'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = self.plot_figure(key, init_run = True)
						)
					], style = {'display': 'flex', 'gap': '10px'})
				)
			
			# Tab 3 - Separatrix
			if self.data_fields[key].category == "Separatrix":
				divs['separatrix'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = self.plot_figure(key, init_run = True)
						)
					], style = {'display': 'flex', 'gap': '10px'}))

			if self.data_fields[key].category == "Biomed":
				divs['special'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = self.plot_figure(key, init_run = True)
						)
					], style = {'display': 'flex', 'gap': '10px'})
				)

		tabs = []
		for key in divs:
			if divs[key] != []:
				tabs.append(dcc.Tab(label = key, children = divs[key]))

		self.app.layout = html.Div(
			className = "container",
			children = [
				html.Div(
					className = "topbar",
					children = [
						dcc.Markdown(
							children = intro_text,
							className = "intro"
						),
						html.Div(
							f"Profile: {self.profile.name}",
							id = "profile-banner",
							className = "profile-badge",
						),
					],
				),

				html.Div(
					className = "main-grid",
					children = [
						html.Div(
							className = "left-panel",
							children = [

								# Controls card
								html.Div(
									className = "card",
									children = [
										html.Div(
											"Controls",
											className = "card-title",
										),

										html.Div(
											className = "control-row",
											children = [
												html.Span(
													"Mode",
													className = "label",
												),
												dcc.RadioItems(
													id = "mode-switch",
													options = [
														{"label": "Live", "value": "live"},
														{"label": "From file", "value": "file"},
													],
													value = "live",
													className = "radio",
												),
											],
										),

										html.Div(
											className = "control-row",
											children = [
												html.Span(
													"Bin length",
													className = "label",
												),
												dcc.Dropdown(
													id = "bin-length",
													options = [
														{"label": str(x), "value": x}
														for x in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
													],
													value = 1,
													clearable = False,
													className = "dropdown small",
												),
											],
										),
									],
								),
								html.Div(
									id = "file-controls",
									className = "card",
									style = {"display": "none"},
									children = [
										html.Div(
											"Load from file",
											className = "card-title",
										),
										dcc.Input(
											id = "file-path",
											type = "text",
											placeholder = "Type a path",
											debounce = False,
											className = "input",
										),
										dcc.Dropdown(
											id = "file-suggest",
											options = [],
											placeholder = "Suggestions",
											searchable = True,
											clearable = True,
											className = "dropdown",
										),
										html.Button(
											"Load file",
											id = "load-file-btn",
											className = "button-primary",
										),
										dcc.Dropdown(
											id = "cycle-selector",
											placeholder = "Select cycle",
											clearable = False,
											searchable = True,
											className = "dropdown small",
										),
										html.Div(
											id = "load-status",
											className = "status-text",
										),
										html.Div(
											id = "cycle-load-trigger",
											style = {"display": "none"},
										),
									],
								),
							],
						),
						html.Div(
							className = "right-panel",
							children = [
								html.Div(
									className = "card main-card",
									children = [
										dcc.Tabs(
											tabs,
											className = "tabs",
										),
									],
								),
							],
						),
					],
				),

				# Hidden infrastructure
				html.Div(
					id = "xaxis-trigger",
					style = {"display": "none"},
				),
				html.Div(
					id = "listener-trigger",
					style = {"display": "none"},
				),

				dcc.Interval(
					id = "refresh",
					interval = 200,
					n_intervals = 0,
				),
			],
		)
		@self.app.callback(
			Output({"type": "stream-graph", "key": MATCH}, "extendData"),
			Input("refresh", "n_intervals"),
			State("mode-switch", "value"),
			State({"type":"stream-graph", "key": MATCH}, "id"),
			State("bin-length", "value"),
		)
		def stream_data(n_intervals, mode, graph_id, bin_length):
			data_key = graph_id["key"]

			df = self.data_fields[data_key]
			bin_info = df.bin

			trace_bufs = df.plot_from or df.buffer_dependance

			trace_indices = []
			xs, ys = [], []

			with self._buflock:
				total = len(self.data_buffer[trace_bufs[0]].data)
				
				if not bin_info or not bin_info['enabled'] or bin_length <= 1:
					ptr = df.buffer_pointer
					if ptr >= total:
						return no_update
					
					end = min(ptr + self.CHUNK_SIZE, total)

					for i, tmp in enumerate(df.plot_order):
						raw_x = self.data_buffer[tmp['x']].data[ptr:end]

						if raw_x and isinstance(raw_x[0], datetime.datetime):
							x_vals = [dt.isoformat() for dt in raw_x]
						else:
							x_vals = raw_x

						try:
							y_vals = [float(y) for y in self.data_buffer[tmp['y']].data[ptr:end]] 
						except TypeError:
							y_vals = [y.value() for y in self.data_buffer[tmp['y']].data[ptr:end]] 
						
						xs.append(x_vals)
						ys.append(y_vals)

						trace_indices.append(i)

					df.buffer_pointer = end

					print(f"Streaming for {graph_id}")
					print(f"\tTraces to stream: {trace_bufs}")
					print(f"\tStreamed {df.buffer_pointer} / {total}")

					return dict(x = xs, y = ys), trace_indices, total

				# binned streaming
				else:
					ptr = getattr(df, "buffer_pointer_bin", 0)
					total = total // bin_length

					if ptr >= total:
						return no_update
					
					end = min(ptr + self.CHUNK_SIZE, total)

					raw_start = ptr * bin_length
					raw_end = end * bin_length

					how_x = bin_info['x']
					how_y = bin_info['y']

					for i, tmp in enumerate(df.plot_order):
						raw_x = self.data_buffer[tmp['x']].data[raw_start:raw_end]
						x_vals = _bin_array(raw_x, bin_length, how_x)

						if x_vals and isinstance(x_vals[0], datetime.datetime):
							x_vals = [dt.isoformat() for dt in x_vals]

						raw_y =  self.data_buffer[tmp['y']].data[raw_start:raw_end]
						y_vals = _bin_array(raw_y, bin_length, how_y)
						
						xs.append(x_vals)
						ys.append(y_vals)

						trace_indices.append(i)

					df.buffer_pointer_bin = end
					print(f"Streaming for {graph_id}")
					print(f"\tTraces to stream: {trace_bufs}")
					print(f"\tStreamed {df.buffer_pointer_bin} / {total}")

					return dict(x = xs, y = ys), trace_indices, total


		@self.app.callback(
			Output("listener-trigger", "children"),
			Input("mode-switch", "value"),
		)
		def _trigger_listener(mode):		
			self._clear_buffer()

			if mode == "live":
				if not getattr(self, "_listener_thread", None):
					self.start_listener()
			else:
				if getattr(self, "_listener_thread", None):
					self.stop_listener()
					print(f"[INFO] Listener terminated.")

					del self._listener_thread
			return ""

		@self.app.callback(
			Output("file-suggest", "options"),
			Input("file-path", "value"),
		)
		def suggest_paths(text):
			if not text:
				return []

			expanded = os.path.expanduser(text)
			if expanded.endswith(os.sep) or Path(expanded).is_dir():
				dirpath = Path(expanded)
				prefix = ""
			else:
				p = Path(expanded)
				dirpath = p.parent if str(p.parent) else Path(".")
				prefix = p.name

			try:
				if not dirpath.exists() or not dirpath.is_dir():
					return []

				pref = prefix.lower()
				entries = []
				for child in dirpath.iterdir():
					name = child.name
					if prefix and not name.lower().startswith(pref):
						continue

					label = name + ("/" if child.is_dir() else "")
					value = str(child) + ("/" if child.is_dir() else "")
					entries.append({"label": label, "value": value})

				entries.sort(key = lambda d: (not d["label"].endswith("/"), d["label"].lower()))
				return entries[:50]

			except PermissionError:
				return [{"label": "Permission denied", "value": ""}]

		@self.app.callback(
			Output("file-path", "value"),
			Input("file-suggest", "value"),
			prevent_initial_call = True,
		)
		def apply_suggestion(selected):
			if not selected:
				return no_update
			return selected

		@self.app.callback(
			Output("file-controls", "style"),
			Input("mode-switch", "value")
		)
		def toggle_file_controls(mode):
			return {"display": "block"} if mode in ["file"] else {"display": "none"}

		@self.app.callback(
			Output("cycle-selector", "options"),
			Output("cycle-selector", "value"),
			Output("load-status", "children"),
			Input("load-file-btn", "n_clicks"),
			State("file-path", "value"),
			State("mode-switch", "value"),
			prevent_initial_call = True
		)
		def load_file_and_populate_cycles(n_clicks, filepath, mode):
			if not filepath or not Path(filepath).is_file():
				return [], None, "No file selected or not found."

			if mode == 'file':
				self.data_from_file = self.profile.read_file(filepath)

				if isinstance(self.data_from_file, pd.DataFrame) and ("cycle_id" in self.data_from_file.columns): 
					unique_cycles = sorted(getattr(self.data_from_file, 'cycle_id').unique())
					options = [{"label": str(c), "value": c} for c in unique_cycles]
					default = unique_cycles[0] if unique_cycles else None
					return options, default, f"Loaded {len(unique_cycles)} cycles."
				else:
					return [0], 0, f"Loaded data."			
			else:
				return [], None, "Unknown error"

		@self.app.callback(
			Output("cycle-load-trigger", "children"),
			Input("mode-switch", "value"),
			Input("cycle-selector", "value"),
			State("file-path", "value"),
			prevent_initial_call = True
		)
		def on_cycle_selected(mode, cycle_id, filepath):
			if cycle_id is None:
				return no_update

			try:
				if mode == "live":
					return no_update

				elif mode == "file":
					self._clear_buffer()

					data_mapping = self.profile.process_file(self, self.data_from_file, cycle_id = cycle_id)

					with self._buflock:
						self.current_batch_id = 0
						for key in data_mapping:
							self.data_buffer[key].extend(data_mapping[key], batch_id = self.current_batch_id)
						
						self.run_callbacks()

				print(f"[INFO] Loaded cycle #{cycle_id}")

			except Exception as e:
				print(f"[ERROR] reading cycle {cycle_id}: {e}")
				traceback.print_exc()
				return no_update

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