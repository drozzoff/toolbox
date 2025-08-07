import dash
from dash import dcc, html, no_update, MATCH
from dash.dependencies import Output, Input, State
from flask_compress import Compress
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import threading
import socket
import json
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from pathlib import Path
import pandas as pd
import time
import datetime


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

	plot_from: list[str] | None = None # list of the buffers that this data field is dependent upon for plotting

	buffer_pointer: int = 0

	plot_order: list[dict] | None = None # Description of the order traces are added to the plot

class TrackingDashboard:
	"""
	Class to manage the tracking dashboard.

	time_coord: str
		Either "time" or "turn". Default is "turn".
	intensity_coord: str
		Either "particles" or "current". Default is "particles".
		
	"""

	CHUNK_SIZE = 5000 # max number points to send
	
	def __init__(
			self,
			host: str = '127.0.0.1',
			port: int = 0,
			data_to_monitor: list[str] | str | None = None
		):

		self.time_coord = 'turn'

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

		self.data_fields = {
			'intensity': DataField(
				buffer_dependance = [self.time_coord, 'Nparticles'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'Nparticles',
						"color": "blue",
						"label": "Intensity in the ring"
					},
				]
			),
			'ES_septum_anode_losses': DataField(
				buffer_dependance = [self.time_coord, 'ES_septum_anode_loss_outside', 'x_extracted_at_ES', 'px_extracted_at_ES'],
				create_new_buffer = ['ES_septum_anode_loss_inside', 'ES_septum_anode_loss_total'],
				callback = self.calculate_total_loss_at_septum,
				plot_from = [self.time_coord, 'ES_septum_anode_loss_inside', 'ES_septum_anode_loss_total']
			),	
			'ES_septum_anode_losses_outside': DataField(
				buffer_dependance = [self.time_coord, 'ES_septum_anode_loss_outside']
			),
			'ES_septum_anode_losses_inside': DataField(
				buffer_dependance = [self.time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES'],
				create_new_buffer = ['ES_septum_anode_loss_inside'],
				callback = self.calculate_loss_inside_septum,
				plot_from = [self.time_coord, 'ES_septum_anode_loss_inside']
			),
			'spill': DataField(
				buffer_dependance = [self.time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES'], 
				create_new_buffer = ['spill'],
				callback = self.calculate_spill,
				plot_from = [self.time_coord, 'spill'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'spill',
						"color": "blue",
						"label": "Spill"
					},
				]
			),
			'spill_mixed': DataField(
				buffer_dependance = [self.time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
				create_new_buffer = ['spill_C', 'spill_He'],
				callback = self.calculate_spill_mixed,
				plot_from = [self.time_coord, 'spill_C', 'spill_He'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'spill_C',
						"color": "blue",
						"label": "Carbon"
					},
					{
						"x": self.time_coord,
						"y": 'spill_He',
						"color": "red",
						"label": "Helium"
					},
				]
			),
			'ES_entrance_phase_space': DataField(
				buffer_dependance = ['x_extracted_at_ES', 'px_extracted_at_ES']
			),
			'MS_entrance_phase_space': DataField(
				buffer_dependance = ['x_extracted_at_MS', 'px_extracted_at_MS']
			),
			'separatrix': DataField(
				buffer_dependance = ['x_stable', 'px_stable', 'x_unstable', 'px_unstable']
			),
			'biomed_data': DataField(
				buffer_dependance = [self.time_coord, 'IC1', 'IC2', 'IC3', 'nozzle'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": "IC2",
						"color": "green",
						"label": "IC2"
					},
					{
						"x": self.time_coord,
						"y": "nozzle",
						"color": "red",
						"label": "nozzle"
					},
					{
						"x": self.time_coord,
						"y": "IC1",
						"color": "blue",
						"label": "IC1"
					},
					{
						"x": self.time_coord,
						"y": "IC3",
						"color": "cyan",
						"label": "IC3"
					},
				]
			)
		}

		self.data_to_expect, self.data_buffer = [], {}

		for data_key in self.data_to_monitor:
			if data_key not in self.data_fields:
				raise ValueError(f"Unsupported data requested: {data_key}. Supported data: {self.data_fields.keys()}")

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

	def _clear_buffer(self):
		# resetting the buffers in the memory
		with self._buflock:
			for key in self.data_buffer:
				self.data_buffer[key].clear()

		# resetting the pointers in the dependent data fields
		for data_key in self.data_fields:
			self.data_fields[data_key].buffer_pointer = 0	

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

									buffers_to_check = self.data_fields[data_key].buffer_dependance
									res_list = [self.data_buffer[key].new_data for key in buffers_to_check]
									
									if all(res_list) and self.data_fields[data_key].callback is not None:
										self.data_fields[data_key].callback()

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

		for i, tmp in enumerate(self.data_fields[key].plot_order):

			x = kwargs.get(tmp['x'], [])
			y = kwargs.get(tmp['y'], [])

			if len(x) != len(y):
				print(f"[ERROR] length missmatch between x and y for '{key}' trace id = {i}")
				return
			
			fig.add_trace(go.Scatter(
				x = x,
				y = y,
				mode = "lines",
				line = dict(
					color = tmp['color']
				),
				name = tmp['label']
			))

		match key:
			case 'intensity': 
				fig.update_layout(
					title = 'Intensity',
					xaxis_title = self.time_coord,
					yaxis_title = 'Intensity [a.u.]',
					width = 1800,
					height = 400,
				)
			
			case 'ES_septum_anode_losses':
				
				fig = make_subplots(
					rows = 1, cols = 2,
					column_widths = [0.8, 0.2],
					horizontal_spacing = 0.05,
					subplot_titles = ["Losses the septum", "Accumulated loss"]
				)

				fig.add_trace(
					go.Scatter(
						x = self.data_buffer[self.time_coord].data,
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
						x = self.data_buffer[self.time_coord].data,
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
						x = self.data_buffer[self.time_coord].data,
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
						x = self.data_buffer[self.time_coord].data,
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
						x = self.data_buffer[self.time_coord].data,
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
						x = self.data_buffer[self.time_coord].data,
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
					xaxis_title = self.time_coord,
					yaxis_title = 'Lost [a.u.]',
					width = 2250,
					height = 400,
				)
				return fig
			
			case 'ES_septum_anode_losses_inside':
				trace = go.Scatter(
					x = self.data_buffer[self.time_coord].data,
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
					xaxis_title = self.time_coord,
					yaxis_title = 'Lost particles [a.u.]',
					width = 1800,
					height = 400,
				)
				return fig
			
			case 'ES_septum_anode_losses_outside':
				trace = go.Scatter(
					x = self.data_buffer[self.time_coord].data,
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
					xaxis_title = self.time_coord,
					yaxis_title = 'Lost particles [a.u.]',
					width = 1800,
					height = 400,
				)
				return fig
		
			case 'spill':
				if self.time_coord == 'time':
					fig.update_xaxes(
						type = "date",
						tickformat = "%H:%M:%S",
						tickangle = 0,
						showgrid = True,
					)

				fig.update_layout(
					title = 'Spill',
					xaxis_title = self.time_coord,
					yaxis_title = 'Spill [a.u.]',
					width = 2250,
					height = 400,
					showlegend = False
				)
			
			case 'spill_mixed':

				if self.time_coord == 'time':
					fig.update_xaxes(
						type = "date",
						tickformat = "%H:%M:%S",
						tickangle = 0,
						showgrid = True,
					)
				
				fig.update_layout(
					title = 'Spill mixed beam',
					xaxis_title = self.time_coord,
					yaxis_title = 'Spill',
					width = 2250,
					height = 900,
				)
			
			case 'biomed_data':
				if self.time_coord == 'time':
					fig.update_xaxes(
						type = "date",
						tickformat = "%H:%M:%S",
						tickangle = 0,
						showgrid = True,
					)
				
				fig.update_layout(
					title = 'Spill, biomed data',
					xaxis_title = self.time_coord,
					yaxis_title = 'Spill',
					width = 2250,
					height = 900,
				)
			
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

		return fig

	def run_dash_server(self):
		"""
		Start a dash server
		"""

		self.app = dash.Dash("Slow extraction w/ xsuite")
		Compress(self.app.server)

		intro_text = '''
		### SIS18 slow extraction dashboard.
		'''

		# Building tabs based on the data requested

		divs = {
			'turn_dependent_data': [],
			'phase_space': [],
			'separatrix': [],
			'special': []
		}
		for key in self.data_to_monitor:
			
			# Tab 1 - Turn dependent data
			if key in {'intensity', 'ES_septum_anode_losses', 'ES_septum_anode_losses_inside', 'ES_septum_anode_losses_outside', 'spill', 'spill_mixed'}:
				divs['turn_dependent_data'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = self.plot_figure(key, init_run = True)
						)
					], style = {'display': 'flex', 'gap': '10px'})
				)
			
			# Tab 2 - Phase space
			if key in {'ES_entrance_phase_space', 'MS_entrance_phase_space'}:
				divs['phase_space'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = self.plot_figure(key, init_run = True)
						)
					], style = {'display': 'flex', 'gap': '10px'})
				)
			
			# Tab 3 - Separatrix
			if key in {'separatrix'}:
				divs['separatrix'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = self.plot_figure(key, init_run = True)
						)
					], style = {'display': 'flex', 'gap': '10px'}))

			if key in {'biomed_data'}:
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


		self.app.layout = html.Div([
			dcc.Markdown(children = intro_text),
			html.Div(
				[
					html.Span("Mode:", style = {"margin-right": "0.5rem", "font-weight": "bold"}),
					dcc.RadioItems(
						id = "mode-switch",
						options = [
							{"label": "Live", "value": "live"},
							{"label": "From file", "value": "file"},
							{"label": "Biomed", "value": "file_biomed"}
						],
						value = "live",
						labelStyle = {"display": "inline-block", "margin-right": "1rem"}
					),
				], style = {"display": "flex", "alignItems": "center", "gap": "0.5rem"}
			),
			html.Div(
				[
					html.Span("Time coordinate:", style = {"margin-right": "0.5rem", "font-weight": "bold"}),
					dcc.RadioItems(
						id = "x-axis-choice",
						options = [
							{"label": "Turn", "value": "turn"},
							{"label": "Time", "value": "time"},
						],
						value = "turn",
						labelStyle = {"display": "inline-block", "margin-right": "1rem"},
					)
				], style = {"display": "flex", "alignItems": "center", "gap": "0.5rem"}
			),
			html.Div(id = "xaxis-trigger", style = {"display": "none"}),
			html.Div([
				dcc.Dropdown(
					id = "file-selector", 
					placeholder = "Select data file",
					searchable = True,
					style = {"width": "250px"}
				),
				html.Button("Load file", id = "load-file-btn"),
				html.Div(id = "load-status"),
				dcc.Dropdown(
					id = "cycle-selector", 
					placeholder = "Select cycle", 
					clearable = False,
					searchable = True,
					style = {"width": "250px"}
				),
				html.Div(id = "cycle-load-trigger", style = {"display": "none"})
			], id = "file-controls", style = {"display": "none"}),
			html.Div(id = "listener-trigger", style = {"display": "none"}),
			dcc.Tabs(tabs),
			dcc.Interval(id = 'refresh', interval = 500, n_intervals = 0)
		])

		@self.app.callback(
			Output({"type": "stream-graph", "key": MATCH}, "extendData"),
			Input("refresh", "n_intervals"),
			State("mode-switch", "value"),
			State({"type":"stream-graph", "key": MATCH}, "id"),
		)
		def stream_data(n_intervals, mode, graph_id):
			data_key = graph_id["key"]

			df = self.data_fields[data_key]
			trace_bufs = df.plot_from or df.buffer_dependance

			trace_indices = []

			xs, ys = [], []

			with self._buflock:
				ptr = df.buffer_pointer
				total = len(self.data_buffer[trace_bufs[0]].data)

				if ptr >= total:
					return no_update
				
				end = min(ptr + self.CHUNK_SIZE, total)

				for i, tmp in enumerate(self.data_fields[data_key].plot_order):
					raw_x = self.data_buffer[tmp['x']].data[ptr:end]

					if raw_x and isinstance(raw_x[0], datetime.datetime):
						x_vals = [dt.isoformat() for dt in raw_x]
					else:
						x_vals = raw_x

					y_vals = [float(y) for y in self.data_buffer[tmp['y']].data[ptr:end]] 
					
					xs.append(x_vals)
					ys.append(y_vals)

					trace_indices.append(i)

					
				df.buffer_pointer = end

			res = dict(x = xs, y = ys)

			return res, trace_indices, total


		@self.app.callback(
			Output("xaxis-trigger", "children"),
			Input("x-axis-choice", "value"),
		)
		def _set_time_coord(choice):
			self.time_coord = choice

			self._clear_buffer()
			self._set_dependencies()

			if not choice in self.data_buffer:
				self.data_buffer[choice] = DataBuffer()

			print(f"[INFO] Set time coordinate to '{choice}'")
			return ""

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
			Output("file-controls", "style"),
			Input("mode-switch", "value")
		)
		def toggle_file_controls(mode):
			return {"display": "block"} if mode in ["file", "file_biomed"] else {"display": "none"}
		
		@self.app.callback(
			Output("file-selector", "options"),
			Input("mode-switch", "value")
		)
		def list_files(mode):
			if mode not in ["file", "file_biomed"]:
				return []
			
			data_dir = Path("data_storage")

			files = sorted(data_dir.glob("*.parquet"))
			return [{"label": f.name, "value": str(f)} for f in files]

		@self.app.callback(
			Output("cycle-selector", "options"),
			Output("cycle-selector", "value"),
			Output("load-status", "children"),
			Input("load-file-btn", "n_clicks"),
			State("file-selector", "value"),
			prevent_initial_call = True
		)
		def load_file_and_populate_cycles(n_clicks, filepath):
			if not filepath or not Path(filepath).is_file():
				return [], None, "No file selected or not found."
			try:
				self.read_from_file = pd.read_parquet(filepath)
			except Exception as e:
				return [], None, f"Read error: {e}"

			if "cycle_id" not in self.read_from_file.columns:
				return [], None, "Missing cycle_id."
			
			print(self.read_from_file)
			unique_cycles = sorted(self.read_from_file["cycle_id"].unique())
			options = [{"label": str(c), "value": c} for c in unique_cycles]
			default = unique_cycles[0] if unique_cycles else None
			return options, default, f"Loaded {len(unique_cycles)} cycles."

		@self.app.callback(
			Output("cycle-load-trigger", "children"),
			Input("mode-switch", "value"),
			Input("cycle-selector", "value"),
			State("file-selector", "value"),
			prevent_initial_call = True
		)
		def on_cycle_selected(mode, cycle_id, filepath):
			if cycle_id is None:
				return no_update

			try:
				if mode == "live":
					return no_update

				elif mode == "file":
					# offline visualization to be included
					pass
					
				elif mode == "file_biomed":
					single_cycle = self.read_from_file[self.read_from_file['cycle_id'] == cycle_id]
					print(single_cycle)

					self._clear_buffer()

					with self._buflock:
						

						self.data_buffer['nozzle'].extend(list(single_cycle['Y[1]'].values))
						self.data_buffer['IC1'].extend(list(single_cycle['Y[0]'].values))
						self.data_buffer['IC2'].extend(list(single_cycle['Y[2]'].values))
						self.data_buffer['IC3'].extend(list(single_cycle['Y[3]'].values))

						self.data_buffer['time'].extend(list(single_cycle.index.to_pydatetime()))


				print(f"[INFO] Loaded cycle #{cycle_id}")

			except Exception as e:
				print(f"[ERROR] reading cycle {cycle_id}: {e}")
				return no_update

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

	test.run_dash_server()