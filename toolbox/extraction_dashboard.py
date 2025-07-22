import dash
from dash import dcc, html, no_update
from dash.dependencies import Output, Input
import plotly.graph_objs as go
from plotly.subplots import make_subplots
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

	__repr__ = __str__

@dataclass
class DataField:
	"""Simple class to store the dependances and state of the data field in the dashboard"""
	buffer_dependance: list[str] = field(default_factory = list)
	create_new_buffer: list[str] | None = None
	callback: Optional[callable] = None
	state: bool = False

class TrackingDashboard:
	"""
	Class to manage the tracking dashboard.
	"""
	def __init__(
			self,
			host: str = '127.0.0.1',
			port: int = 0,
			data_to_monitor: list[str] | str | None = None
		):
		self.data_fields = {
			'intensity': DataField(['turn', 'Nparticles']),
			'ES_septum_anode_losses': DataField(
				['turn', 'ES_septum_anode_loss_outside', 'x_extracted_at_ES', 'px_extracted_at_ES'],
				['ES_septum_anode_loss_inside', 'ES_septum_anode_loss_total'],
				self.calculate_total_loss_at_septum
			),	
			'ES_septum_anode_losses_outside': DataField(['turn', 'ES_septum_anode_loss_outside']),
			'ES_septum_anode_losses_inside': DataField(
				['turn', 'x_extracted_at_ES', 'px_extracted_at_ES'],
				['ES_septum_anode_loss_inside'],
				self.calculate_loss_inside_septum
			),
			'spill': DataField(
				['turn', 'x_extracted_at_ES', 'px_extracted_at_ES'], 
				['spill'],
				self.calculate_spill
			),
			'spill_mixed': DataField(
				['turn', 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
				['spill_C', 'spill_He'],
				self.calculate_spill_mixed
			),
			'ES_entrance_phase_space': DataField(['x_extracted_at_ES', 'px_extracted_at_ES']),
			'MS_entrance_phase_space': DataField(['x_extracted_at_MS', 'px_extracted_at_MS']),
			'separatrix': DataField(['x_stable', 'px_stable', 'x_unstable', 'px_unstable'])
		}

		self._buflock = threading.Lock()

		if host is None:
			raise ValueError("Host cannot be `None`.")
		
		if data_to_monitor is None:
			raise ValueError("No data to monitor provided")
		
		self.host, self.port = host, port
		self.data_to_monitor = data_to_monitor

		if isinstance(data_to_monitor, str):
			self.data_to_monitor = [self.data_to_monitor]

		self.data_to_expect, self.data_buffer = [], {}

		for data_key in self.data_to_monitor:
			if data_key not in self.data_fields:
				raise ValueError(f"Unsupported data requested: {data_key}. Supported data: {self.data_fields}")

			# if needed creating buffers for data to read from the user
			for key in self.data_fields[data_key].buffer_dependance:
				if not key in self.data_to_expect:
					self.data_to_expect.append(key)
			
					self.data_buffer[key] = DataBuffer()
			
			# creating additional buffers
			if self.data_fields[data_key].create_new_buffer is not None:
				for key in self.data_fields[data_key].create_new_buffer:
					self.data_buffer[key] = DataBuffer()

			# activating the DataField
			self.data_fields[data_key].state = True

	def calculate_loss_inside_septum(self, append_to_buffer = True): 		
		x = np.array(self.data_buffer['x_extracted_at_ES'].recent_data)
		px = np.array(self.data_buffer['px_extracted_at_ES'].recent_data)

		threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
		lost_inside_septum_mask = x > threshold
		
		if append_to_buffer:
			self.data_buffer['ES_septum_anode_loss_inside'].append(sum(lost_inside_septum_mask))
		
		return lost_inside_septum_mask

	def calculate_total_loss_at_septum(self):
		# the bot check is solely for the case where we plot the total loss including the other 
		# losses aas well
		append_to_buffer = False if 'ES_septum_anode_losses_inside' in self.data_to_monitor else True
		res = sum(self.calculate_loss_inside_septum(append_to_buffer = append_to_buffer)) + self.data_buffer['ES_septum_anode_loss_outside'].recent_data[0]

		self.data_buffer['ES_septum_anode_loss_total'].append(res)

	def calculate_spill(self):
		extracted = len(self.data_buffer['x_extracted_at_ES'].recent_data)
		lost_inside = sum(self.calculate_loss_inside_septum(append_to_buffer = False))
#		print(f"Extracted = {extracted}, Lost inside = {lost_inside}")
		self.data_buffer['spill'].append(extracted - lost_inside)

	def calculate_spill_mixed(self):
		lost_inside = self.calculate_loss_inside_septum(append_to_buffer = False)

		# separating Carbon from Helium
		ion = np.array(self.data_buffer['ion'].recent_data)
		is_C = ion == "carbon"
		is_He = ion == "helium"

		extracted_C = is_C & ~lost_inside
		extracted_He = is_He & ~lost_inside

		self.data_buffer['spill_C'].append(sum(extracted_C))
		self.data_buffer['spill_He'].append(sum(extracted_He))


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

							with self._buflock:
								for key in self.data_to_expect:
									if key in incoming:
										self.data_buffer[key].extend(incoming[key])
								
								for data_key in self.data_to_monitor:
									# running callback on the data_key only when there is new data 
									# in all the dependant data buffers
									if self.update_figure(data_key) and self.data_fields[data_key].callback is not None:
										self.data_fields[data_key].callback()

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
		res_list = [self.data_buffer[key].new_data for key in self.data_fields[data_key].buffer_dependance]
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
				
				fig = make_subplots(
					rows = 1, cols = 2,
					column_widths = [0.8, 0.2],
					horizontal_spacing = 0.05,
					subplot_titles = ["Losses the septum", "Accumulated loss"]
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = self.data_buffer['ES_septum_anode_loss_total'].data,
						mode = 'lines',
						line = {
							'color': 'green',
							'width': 2,
						},
						name = "Total losses the septum",
						showlegend = True
					),
					row = 1,
					col = 1
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = self.data_buffer['ES_septum_anode_loss_outside'].data,
						mode = 'lines',
						line = {
							'color': 'red',
							'width': 2,
						},
						name = "Lost outside of the septum",
						showlegend = True
					),
					row = 1,
					col = 1
				)
				
				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = self.data_buffer['ES_septum_anode_loss_inside'].data,
						mode = 'lines',
						line = {
							'color': 'blue',
							'width': 2,
						},
						name = "Lost inside of the septum",
						showlegend = True
					),
					row = 1,
					col = 1
				)
				
				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = np.cumsum(self.data_buffer['ES_septum_anode_loss_total'].data),
						mode = 'lines',
						line = {
							'color': 'green',
							'width': 2,
						},
						name = "Total losses the septum",
						showlegend = False
					),
					row = 1,
					col = 2
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = np.cumsum(self.data_buffer['ES_septum_anode_loss_outside'].data),
						mode = 'lines',
						line = {
							'color': 'red',
							'width': 2,
						},
						name = "Lost outside of the septum",
						showlegend = False
					),
					row = 1,
					col = 2
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = np.cumsum(self.data_buffer['ES_septum_anode_loss_inside'].data),
						mode = 'lines',
						line = {
							'color': 'blue',
							'width': 2,
						},
						name = "Lost inside of the septum",
						showlegend = False
					),
					row = 1,
					col = 2
				)
				
				fig.update_layout(
					title = 'Es losses on the anode',
					xaxis_title = 'Turn',
					yaxis_title = 'Number of lost particles',
					width = 2250,
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
				fig = make_subplots(
					rows = 1, cols = 2,
					column_widths = [0.8, 0.2],
					horizontal_spacing = 0.05,
					subplot_titles = ["Spill", "Extracted"]
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = self.data_buffer['spill'].data, 
						mode = 'lines',
						line = dict(
							color = "blue"
						),
						name = "Spill"
					),
					row = 1, 
					col = 1
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = np.cumsum(self.data_buffer['spill'].data), 
						mode = 'lines',
						line = dict(
							color = "blue"
						),
						name = "Extracted"
					),
					row = 1, 
					col = 2
				)

				fig.update_layout(
					title = 'Spill',
					xaxis_title = 'Turn',
					yaxis_title = 'Number of particles',
					width = 2250,
					height = 400,
					showlegend = False
				)
				return fig
			
			case 'spill_mixed':
				cumsum_c = np.cumsum(self.data_buffer['spill_C'].data)
				cumsum_he = np.cumsum(self.data_buffer['spill_He'].data)
				
				spill_c = np.array(self.data_buffer['spill_C'].data)
				spill_he = np.array(self.data_buffer['spill_He'].data)

				fig = make_subplots(
					rows = 2, cols = 2,
					column_widths = [0.8, 0.2],
					horizontal_spacing = 0.05,
					subplot_titles = ["Spill", "Extracted", "Spill He/C"]
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = self.data_buffer['spill_C'].data, 
						mode = 'lines',
						line = dict(
							color = "blue"
						),
						name = "Carbon",
					),
					row = 1,
					col = 1
				)
				
				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = self.data_buffer['spill_He'].data, 
						mode = 'lines',
						line = dict(
							color = "red"
						),
						name = "Helium",
					),
					row = 1,
					col = 1
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = cumsum_c, 
						mode = 'lines',
						line = dict(
							color = "blue"
						),
						name = "Carbon",
					),
					row = 1,
					col = 2
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = cumsum_he, 
						mode = 'lines',
						line = dict(
							color = "red"
						),
						name = "Helium",
					),
					row = 1,
					col = 2
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer['turn'].data,
						y = spill_he - spill_c, 
						mode = 'lines',
						line = dict(
							color = "green"
						),
						name = "Spill He - Spill C",
					),
					row = 2,
					col = 1
				)


				fig.update_layout(
					title = 'Spill mixed beam',
					xaxis_title = 'Turn',
					yaxis_title = 'Number of particles',
					width = 2250,
					height = 900,
				)

#				fig.update_yaxes(range = [-0.5, 3], row = 2, col = 1)

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
					x = self.data_buffer['x_unstable'].data,
					y = self.data_buffer['px_unstable'].data,
					mode = 'markers',
					marker = dict(
						size = 5,
						color = 'red',
					),
					name = "Unstable particle",
					showlegend = True
				)

				trace2 = go.Scatter(
					x = self.data_buffer['x_stable'].data,
					y = self.data_buffer['px_stable'].data,
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
			if key in {'intensity', 'ES_septum_anode_losses', 'ES_septum_anode_losses_inside', 'ES_septum_anode_losses_outside', 'spill', 'spill_mixed'}:
				divs['turn_dependent_data'].append(html.Div([dcc.Graph(id = key)], style = {'display': 'flex', 'gap': '10px'}))
			
			# Tab 2 - Phase space
			if key in {'ES_entrance_phase_space', 'MS_entrance_phase_space'}:
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
			with self._buflock:
				for data_key in self.data_to_monitor:
					if self.update_figure(data_key):
						updates.append(self.plot_figure(data_key))
					else:
						updates.append(no_update)
				
				for data_key in self.data_to_monitor:
					for key in self.data_fields[data_key].buffer_dependance:
						self.data_buffer[key].new_data = False
						self.data_buffer[key].recent_data = []

			return updates
		
#		print("LAYOUT children for phase_space tab:", self.app.layout.children[1].children)

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
#			"spill",
			"spill_mixed",
			"ES_entrance_phase_space",
#			"MS_entrance_phase_space",
			"separatrix"
		]
	)
	test.start_listener()

	test.run_dash_server()