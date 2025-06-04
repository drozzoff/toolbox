import dash
from dash import dcc, html, no_update
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import threading
import socket
import json
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Any


class DataBuffer:
	"""
	`new_data` stays `True` after the data is appended and turns `False` when it is plotted
	`recent_data` is reset to `[]` when the data is plotted
	"""
	def __init__(self):
		self.data, self.recent_data = [], []
		self.new_data = False

	def append(self, value):
		self.recent_data = [value]
		self.data.append(value)
		self.new_data = True

	def extend(self, values):
		self.recent_data = values
		self.data.extend(values)
		self.new_data = True

	def clear(self):
		self.data.clear()
		self.recent_data.clear()
		self.new_data = False

	def __str__(self):
		res = f"data = {self.data}\n"
		res += f"Received data since last update = {self.new_data}\n"
		if self.new_data:
			res += f"New data received = {self.recent_data}\n"
		return res


class TrackingDashboard:
	"""
	Class to manage the tracking dashboard.
	"""
	_supported_data = { 
		'intensity': {
			'_dep': ['turn', 'Nparticles'],
			'_new': None,
			'_callback': None
			},
		'ES_septum_anode_losses': {
			'_dep': ['turn', 'ES_septum_anode_loss_outside', 'x_extracted_at_ES', 'px_extracted_at_ES'],
			'_new':  ['ES_septum_anode_loss_inside', 'ES_septum_anode_loss_total'],
			'_callback': None
			},
		'ES_septum_anode_losses_outside': {
			'_dep': ['turn', 'ES_septum_anode_loss_outside'],
			'_new': None,
			'_callback': None
		},
		'ES_septum_anode_losses_inside': {
			'_dep': ['turn', 'x_extracted_at_ES', 'px_extracted_at_ES'],
			'_new': ['ES_septum_anode_loss_inside'],
			'_callback': None
		},
		'spill': {
			'_dep': ['turn', 'x_extracted_at_ES', 'px_extracted_at_ES'],
			'_new': ['spill'],
			'_callback': None
		},
		'ES_entrance_phase_space': {
			'_dep': ['x_extracted_at_ES', 'px_extracted_at_ES'],
			'_new': None,
			'_callback': None,
		},
		'MS_entrance_phase_space': {
			'_dep': ['x_extracted_at_MS', 'px_extracted_at_MS'],
			'_new': None,
			'_callback': None,
		},
		'separatrix': {
			'_dep': ['x_stable', 'px_stable', 'x_unstable', 'px_unstable'],
			'_new': None,
			'_callback': None,
		}
	}

	def __init__(
			self,
			host: str = '127.0.0.1',
			port: int = 0,
			data_to_monitor: list[str] | str | None = None
		):
		"""
		"""

		if host is None:
			raise ValueError("Host cannot be `None`.")
		
		if data_to_monitor is None:
			raise ValueError("No data to monitor provided")
		
		self.host, self.port = host, port
		self.data_to_monitor = data_to_monitor

		if isinstance(data_to_monitor, str):
			self.data_to_monitor = [self.data_to_monitor]

		self.data_to_expect, self.data_buffer = [], {}

		print(self.data_to_monitor)
		for data_key in self.data_to_monitor:
			if data_key not in self._supported_data:
				raise ValueError(f"Unsupported data requested: {data_key}. Supported data: {self._supported_data}")

			for key in self._supported_data[data_key]['_dep']:
				if not key in self.data_to_expect:
					self.data_to_expect.append(key)
			
					self.data_buffer[key] = DataBuffer()
			
			if self._supported_data[data_key]['_new'] is not None:
				for key in self._supported_data[data_key]['_new']:
					self.data_buffer[key] = DataBuffer()

		self._assign_callbacks()

	def calculate_loss_inside_septum(self, append_to_buffer = True) -> int: 		
		res = 0
		for x, px in zip(self.data_buffer['x_extracted_at_ES'].recent_data, self.data_buffer['px_extracted_at_ES'].recent_data):
			if x > -0.055 - (px + 7.4e-3)**2/(2 * 1.7857e-3):
				res += 1
		
		if append_to_buffer:
			self.data_buffer['ES_septum_anode_loss_inside'].append(res)
		
		return res

	def calculate_total_loss_at_septum(self):
		# the bot check is solely for the case where we plot the total loss including the other 
		# losses aas well
		append_to_buffer = False if 'ES_septum_anode_losses_inside' in self.data_to_monitor else True
		res = self.calculate_loss_inside_septum(append_to_buffer = append_to_buffer) + self.data_buffer['ES_septum_anode_loss_outside'].recent_data[0]

		self.data_buffer['ES_septum_anode_loss_total'].append(res)

	def calcualte_spill(self):
		extracted = len(self.data_buffer['x_extracted_at_ES'].recent_data)
		lost_inside = self.calculate_loss_inside_septum(append_to_buffer = False)
#		print(f"Extracted = {extracted}, Lost inside = {lost_inside}")
		self.data_buffer['spill'].append(extracted - lost_inside)

	def _assign_callbacks(self):
		self._supported_data['ES_septum_anode_losses_inside']['_callback'] = self.calculate_loss_inside_septum

		self._supported_data['ES_septum_anode_losses']['_callback'] = self.calculate_total_loss_at_septum

		self._supported_data['spill']['_callback'] = self.calcualte_spill

	def start_listener(self):
		def run():
			srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			try:
				srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
			except AttributeError:
				pass

			srv.bind((self.host, self.port))
			srv.listen(1)

			assigned_port = srv.getsockname()[1]
			print(f"[INFO] Listening on port {assigned_port}")
			
			while True:
				conn, addr = srv.accept()
				print(f"[INFO] Connection from {addr}")
				for key in self.data_buffer:
					self.data_buffer[key].clear()

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
#							print(incoming)
							for key in self.data_to_expect:
								if key in incoming:
									self.data_buffer[key].extend(incoming[key])
							
							for key_data in self.data_to_monitor:
								if self._supported_data[key_data]['_callback'] is not None:
									self._supported_data[key_data]['_callback']()

				except json.JSONDecodeError as e:
					print("[ERROR] Invalid JSON", e)
				finally:
					conn.close()
					print("[INFO] Client disconnected, back to listening")
		
		threading.Thread(target = run, daemon = True).start()
	
	def update_figure(self, data_key):
		'''
		If new data available - returns `True` to update the figure.
		'''

		res_list = [self.data_buffer[key].new_data for key in self._supported_data[data_key]['_dep']]
		if not(all(res_list) or not any(res_list)):
			raise ValueError(f"Mismatch between the incoming data for {data_key}")
		
		if all(res_list):
			return True
		
		return False

	def plot_figure(self, key):
		match key:
			case 'intensity':
				fig = go.Figure(
					data = go.Scatter(
						x = self.data_buffer['turn'].data,
						y = self.data_buffer['Nparticles'].data, 
						mode = 'lines'
					)
				)
				fig.update_layout(
					title = 'Intensity',
					xaxis_title = 'Turn',
					yaxis_title = 'Number of particles in the ring',
					width = 1800,
					height = 400,
				)
				return fig
			
			case 'ES_septum_anode_losses':
				
				trace3 = go.Scatter(
					x = self.data_buffer['turn'].data,
					y = self.data_buffer['ES_septum_anode_loss_inside'].data,
					mode = 'lines',
					line = {
						'color': 'blue',
						'width': 2,
					},
					name = "Lost inside of the septum",
					showlegend = True
				)

				trace2 = go.Scatter(
					x = self.data_buffer['turn'].data,
					y = self.data_buffer['ES_septum_anode_loss_outside'].data,
					mode = 'lines',
					line = {
						'color': 'red',
						'width': 2,
					},
					name = "Lost outside of the septum",
					showlegend = True
				)
				
				trace1 = go.Scatter(
					x = self.data_buffer['turn'].data,
					y = self.data_buffer['ES_septum_anode_loss_total'].data,
					mode = 'lines',
					line = {
						'color': 'green',
						'width': 2,
					},
					name = "Total losses the septum",
					showlegend = True
				)
				
				fig = go.Figure(data = [trace1, trace2, trace3])
				
				fig.update_layout(
					title = 'Es losses on the anode',
					xaxis_title = 'Turn',
					yaxis_title = 'Number of lost particles',
					width = 1800,
					height = 400,
				)
				return fig
			
			case 'ES_septum_anode_losses_inside':
				trace = go.Scatter(
					x = self.data_buffer['turn'].data,
					y = self.data_buffer['ES_septum_anode_loss_inside'].data,
					mode = 'lines',
					line = {
						'color': 'blue',
						'width': 2,
					},
					name = "Lost inside of the septum",
					showlegend = True
				)
				
				fig = go.Figure(data = [trace])
				
				fig.update_layout(
					title = 'Es losses on the inside of anode',
					xaxis_title = 'Turn',
					yaxis_title = 'Number of lost particles',
					width = 1800,
					height = 400,
				)
				return fig
			
			case 'ES_septum_anode_losses_outside':
				trace = go.Scatter(
					x = self.data_buffer['turn'].data,
					y = self.data_buffer['ES_septum_anode_loss_outside'].data,
					mode = 'lines',
					line = {
						'color': 'blue',
						'width': 2,
					},
					name = "Lost outside of the septum",
					showlegend = True
				)
				
				fig = go.Figure(data = [trace])
				
				fig.update_layout(
					title = 'Es losses on the outside of anode',
					xaxis_title = 'Turn',
					yaxis_title = 'Number of lost particles',
					width = 1800,
					height = 400,
				)
				return fig
		
			case 'spill':
				fig = go.Figure(
					data = go.Scatter(
						x = self.data_buffer['turn'].data,
						y = self.data_buffer['spill'].data, 
						mode = 'lines'
					)
				)
				fig.update_layout(
					title = 'Spill',
					xaxis_title = 'Turn',
					yaxis_title = 'Number of particles in the ring',
					width = 1800,
					height = 400,
				)
				return fig
			
			case 'ES_entrance_phase_space':
				phase_space_fig = go.Figure(
					data = go.Scatter(
						x = self.data_buffer['x_extracted_at_ES'].data, 
						y = self.data_buffer['px_extracted_at_ES'].data, 
						mode = 'markers',
						marker = dict(
							size = 5,
							color = 'black',
							opacity = 0.3
						),
						name = "Extracted particles"
					)
				)

				# Anode
				phase_space_fig.add_shape(
					type = 'line',
					x0 = -0.055, y0 = -0.0085,
					x1 = -0.055, y1 = -0.005,
					line = dict(
						color = "LightSeaGreen",
						width = 4,
						dash = "dashdot",
					),
					name = "Anode",
					showlegend = True
				)

				# Cathode
				cathode_x_line = [-0.073, -0.073, -0.083, -0.083]
				cathode_y_line = [-0.0085, -0.005, -0.005, -0.0085]

				phase_space_fig.add_trace(go.Scatter(
					x = cathode_x_line,
					y = cathode_y_line,
					fill = 'toself',
					fillcolor = 'rgba(0, 0, 255, 0.3)',
					line = dict(color = 'rgba(0, 0, 0, 0)'),
					name = "Cathode",
				))

				# limits on ont being lost inside of the septum
				
				px_loss_limit = np.linspace(-7.4e-3, -5.0e-3, 100).tolist()
				x_loss_limit = list(map(lambda px: -0.055 - (px + 7.4e-3)**2 / (2 * 1.7857e-3), px_loss_limit))

				phase_space_fig.add_trace(go.Scatter(
					x = x_loss_limit,
					y = px_loss_limit,
					mode = 'lines',
					line = dict(
						color = 'red',
						dash = 'dash',
						),
					name = "Lost inside on the wires limit",
					showlegend = True
				))


				phase_space_fig.update_layout(
					title = 'Phase space at ES entrance',
					width = 800,
					height = 700,
					xaxis_title = 'x [m]',
					yaxis_title = 'px [rad]',
					showlegend = True
				)
				return phase_space_fig
			
			case 'MS_entrance_phase_space':
				phase_space_fig = go.Figure(
					data = go.Scatter(
						x = self.data_buffer['x_extracted_at_MS'].data, 
						y = self.data_buffer['px_extracted_at_MS'].data, 
						mode = 'markers',
						marker = dict(
							size = 5,
							color = 'black',
							opacity = 0.3
						),
						name = "Extracted particles"
					)
				)

				phase_space_fig.add_shape(
					type = 'line',
					x0 = 0.038, y0 = 0.003,
					x1 = 0.038, y1 = 0.009,
					line = dict(
						color = "LightSeaGreen",
						width = 4,
						dash = "dashdot",
					),
					name = "Bottom of the septum",
					showlegend = True
				)

				phase_space_fig.add_shape(
					type = 'line',
					x0 = 0.1, y0 = 0.003,
					x1 = 0.1, y1 = 0.009,
					line = dict(
						color = "SeaGreen",
						width = 4,
						dash = "dash",
					),
					name = "Top of the septum",
					showlegend = True
				)

				phase_space_fig.add_trace(go.Scatter(
					x = [0.069],
					y = [8e-3],
					mode = 'markers',
					name = 'Septum centeer orbit',
					marker = dict(
						symbol = 'x',
						size = 12,
						color = 'red',
						line = dict(width = 2)
					),
					showlegend = True
				))

				phase_space_fig.update_layout(
					title = 'Phase space at MS entrance',
					width = 800,
					height = 700,
					xaxis_title = 'x [m]',
					yaxis_title = 'px [rad]',
					showlegend = True
				)
				return phase_space_fig

			case 'separatrix':
				trace1 = go.Scatter(
					x = self.data_buffer['x_stable'],
					y = self.data_buffer['px_stable'],
					mode = 'markers',
					marker = dict(
						size = 5,
						color = 'red',
					),
					name = "Unstable particle",
					showlegend = True
				)

				trace2 = go.Scatter(
					x = self.data_buffer['x_unstable'],
					y = self.data_buffer['px_unstable'],
					mode = 'markers',
					marker = dict(
						size = 5,
						color = 'green',
					),
					name = "Stable particle",
					showlegend = True
				)
				
				sep_fig = go.Figure(data = [trace1, trace2])

				sep_fig.add_shape(
					type = 'line',
					x0 = -0.055, y0 = -0.0085,
					x1 = -0.055, y1 = 0.005,
					line = dict(
						color = "SeaGreen",
						width = 4,
						dash = "dash",
					),
					name = "ES Septum anode",
					showlegend = True
				)

				sep_fig.update_layout(
					title = 'Separatrix',
					width = 1000,
					height = 800,
					xaxis_title = 'x [m]',
					yaxis_title = 'px [rad]',
					showlegend = True
				)

				return sep_fig

	def run_dash_server(self):
		"""
		Start a dash server
		"""

		self.app = dash.Dash("Slow extraction w/ xsuite")

		intro_text = '''
		### SIS18 slow extraction dashboard.
		'''

		# Building tabs based on the data requested

		divs = {
			'turn_dependent_data': [],
			'phase_space': [],
			'separatrix': []
		}
		for key in self.data_to_monitor:
			
			# Tab 1 - Turn dependent data
			if key in {'intensity', 'ES_septum_anode_losses', 'ES_septum_anode_losses_inside', 'ES_septum_anode_losses_outside', 'spill'}:
				divs['turn_dependent_data'].append(html.Div([dcc.Graph(id = key)], style = {'display': 'flex', 'gap': '10px'}))
			
			# Tab 2 - Phase space
			if key in {'ES_entrance_phase_space', 'NS_entrance_phase_space'}:
				divs['phase_space'].append(html.Div([dcc.Graph(id = key)], style = {'display': 'flex', 'gap': '10px'}))
			
			# Tab 3 - Separatrix
			if key in {'separatrix'}:
				divs['separatrix'].append(html.Div([dcc.Graph(id = key)], style = {'display': 'flex', 'gap': '10px'}))

		tabs = []
		for key in divs:
			if divs[key] != []:
				tabs.append(dcc.Tab(label = key, children = divs[key]))


		self.app.layout = html.Div([
			dcc.Markdown(children = intro_text),
			dcc.Tabs(tabs),
			dcc.Interval(id = 'refresh', interval = 500, n_intervals = 0)
		])

		callback_outputs = [Output(x, 'figure') for x in self.data_to_monitor]

		@self.app.callback(callback_outputs, [Input('refresh', 'n_intervals')])
		def update_graph(n):

			updates = []
			for data_key in self.data_to_monitor:
				if self.update_figure(data_key):
					updates.append(self.plot_figure(data_key))
				else:
					updates.append(no_update)
			
			for data_key in self.data_to_monitor:
				for key in self._supported_data[data_key]['_dep']:
					self.data_buffer[key].new_data = False
					self.data_buffer[key].recent_data = []

			return updates

		try:
			print("[INFO] Starting Dash server...")
			self.app.run(debug = True, use_reloader = False)
		except KeyboardInterrupt:
			print("\n[INFO] Caught Ctrl+C. Cleaning up...")
			sys.exit(0)

if __name__ == "__main__":
	test = TrackingDashboard(
		port = 35235, 
		data_to_monitor = [
			"intensity", 
			"ES_septum_anode_losses", 
			"spill", 
			"ES_entrance_phase_space"
		]
	)
	test.start_listener()

	test.run_dash_server()