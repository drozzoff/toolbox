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
import pickle as pk
import datetime
import xtrack as xt


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
						"settings": dict(
							mode = "lines",
							line = {"color": "blue"},
							name = "Intensity in the ring",
							showlegend = True
						)
					},
				]
			),
			'ES_septum_anode_losses': DataField(
				buffer_dependance = [self.time_coord, 'ES_septum_anode_loss_outside', 'x_extracted_at_ES', 'px_extracted_at_ES'],
				create_new_buffer = ['ES_septum_anode_loss_inside', 'ES_septum_anode_loss_total'],
				callback = self.calculate_total_loss_at_septum,
				plot_from = [self.time_coord, 'ES_septum_anode_loss_inside', 'ES_septum_anode_loss_total', 'ES_septum_anode_loss_outside'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_total',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'green',
								'width': 2,
							},
							name = "Total losses",
							showlegend = True
						)
					},
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_outside',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'red',
								'width': 2,
							},
							name = "Lost outside of the septum",
							showlegend = True
						)
					},
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_inside',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'blue',
								'width': 2,
							},
							name = "Lost inside of the septum",
							showlegend = True
						)
					}

				]
			),
			'ES_septum_anode_losses_accumulated': DataField(
				buffer_dependance = [self.time_coord, 'ES_septum_anode_loss_outside', 'x_extracted_at_ES', 'px_extracted_at_ES'],
				create_new_buffer = ['ES_septum_anode_loss_outside_accumulated', 'ES_septum_anode_loss_inside_accumulated', 'ES_septum_anode_loss_total_accumulated'],
				callback = self.calculate_total_accumulated_loss_at_septum,
				plot_from = [self.time_coord, 'ES_septum_anode_loss_outside_accumulated', 'ES_septum_anode_loss_inside_accumulated', 'ES_septum_anode_loss_total_accumulated'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_total_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'green',
								'width': 2,
							},
							name = "Total losses",
							showlegend = True
						)
					},
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_outside_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'red',
								'width': 2,
							},
							name = "Lost outside",
							showlegend = True
						)
					},
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_inside_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'blue',
								'width': 2,
							},
							name = "Lost inside",
							showlegend = True
						)
					}

				]
			),
			'ES_septum_anode_losses_mixed_accumulated': DataField(
				buffer_dependance = [
					self.time_coord, 
					'ES_septum_anode_loss_outside_C', 
					'ES_septum_anode_loss_outside_He', 
					'x_extracted_at_ES', 
					'px_extracted_at_ES'
				],
				create_new_buffer = [
					'ES_septum_anode_loss_outside_C_accumulated',
					'ES_septum_anode_loss_outside_He_accumulated',
					'ES_septum_anode_loss_inside_C_accumulated',
					'ES_septum_anode_loss_inside_He_accumulated',
					'ES_septum_anode_loss_total_C_accumulated',
					'ES_septum_anode_loss_total_He_accumulated',
					'ES_septum_anode_loss_total_accumulated'
				],
				callback = self.calculate_total_accumulated_loss_at_septum_mixed,
				plot_from = [
					self.time_coord, 
					'ES_septum_anode_loss_outside_C_accumulated',
					'ES_septum_anode_loss_outside_He_accumulated',
					'ES_septum_anode_loss_inside_C_accumulated',
					'ES_septum_anode_loss_inside_He_accumulated',
					'ES_septum_anode_loss_total_C_accumulated',
					'ES_septum_anode_loss_total_He_accumulated',
					'ES_septum_anode_loss_total_accumulated'
				],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_total_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'green',
								'width': 2,
							},
							name = "Total losses",
							showlegend = True
						)
					},
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_total_C_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'dash': "dash",
								'color': 'green',
								'width': 2,
							},
							name = "Total losses (C)",
							showlegend = True
						)
					},
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_total_He_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'dash': "dashdot",
								'color': 'green',
								'width': 2,
							},
							name = "Total losses (He)",
							showlegend = True
						)
					},
					
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_outside_C_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'dash': 'dash',
								'color': 'red',
								'width': 2,
							},
							name = "Lost outside (C)",
							showlegend = True
						)
					},

					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_outside_He_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'dash': 'dashdot',
								'color': 'red',
								'width': 2,
							},
							name = "Lost outside (He)",
							showlegend = True
						)
					},

					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_inside_C_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'dash': 'dash',
								'color': 'blue',
								'width': 2,
							},
							name = "Lost inside (C)",
							showlegend = True
						)
					},

					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_inside_He_accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'dash': 'dashdot',
								'color': 'blue',
								'width': 2,
							},
							name = "Lost inside (He)",
							showlegend = True
						)
					},

				]
			),
			'ES_septum_anode_losses_outside': DataField(
				buffer_dependance = [self.time_coord, 'ES_septum_anode_loss_outside'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_outside',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'blue',
								'width': 2,
							},
							name = "Lost outside of the septum",
							showlegend = True
						)
					}
				]
			),
			'ES_septum_anode_losses_inside': DataField(
				buffer_dependance = [self.time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES'],
				create_new_buffer = ['ES_septum_anode_loss_inside'],
				callback = self.calculate_loss_inside_septum,
				plot_from = [self.time_coord, 'ES_septum_anode_loss_inside'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'ES_septum_anode_loss_inside',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'blue',
								'width': 2,
							},
							name = "Lost inside of the septum",
							showlegend = True
						)
					}
				]
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
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "blue"
							),
							name = "Spill"
						)
					},
				]
			),
			'spill_accumulated': DataField(
				buffer_dependance = [self.time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES'], 
				create_new_buffer = ['spill_accumulated'],
				callback = self.calcualte_spill_accumulated,
				plot_from = [self.time_coord, 'spill_accumulated'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'spill_accumulated',
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "blue"
							),
							name = "Spill"
						)
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
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "blue"
							),
							name = "Carbon"
						)
					},
					{
						"x": self.time_coord,
						"y": 'spill_He',
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "red"
							),
							name = "Helium"
						)
					},
				]
			),
			'spill_mixed_integrated': DataField(
				buffer_dependance = [self.time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
				create_new_buffer = ['_spill_C_accumulated', '_spill_He_accumulated', 'spill_C_integrated', 'spill_He_integrated'],
				callback = self.calculate_spill_mixed_integrated,
				plot_from = [self.time_coord, 'spill_C_integrated', 'spill_He_integrated'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'spill_C_integrated',
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "blue"
							),
							name = "Carbon"
						)
					},
					{
						"x": self.time_coord,
						"y": 'spill_He_integrated',
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "red"
							),
							name = "Helium"
						)
					},
				]
			),
			'spill_mixed_accumulated': DataField(
				buffer_dependance = [self.time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
				create_new_buffer = ['spill_C_accumulated', 'spill_He_accumulated'],
				callback = self.calculate_spill_mixed_accumulated,
				plot_from = [self.time_coord, 'spill_C_accumulated', 'spill_He_accumulated'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'spill_C_accumulated',
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "blue"
							),
							name = "Carbon"
						)
					},
					{
						"x": self.time_coord,
						"y": 'spill_He_accumulated',
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "red"
							),
							name = "Helium"
						)
					},
				]
			),
			'spill_mixed_diff_accumulated': DataField(
				buffer_dependance = [self.time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
				create_new_buffer = ['He_C_difference_accumulated', '_C_accumulated', '_He_accumulated'],
				callback = self.calculate_spill_mixed_diff_accumulated,
				plot_from = [self.time_coord, 'He_C_difference_accumulated'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": 'He_C_difference_accumulated',
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "green"
							),
							name = "Helium / Carbon"
						)
					},
				]
			),
			'ES_entrance_phase_space': DataField(
				buffer_dependance = ['x_extracted_at_ES', 'px_extracted_at_ES'],
				plot_order = [
					{
						"x": 'x_extracted_at_ES',
						"y": 'px_extracted_at_ES',
						"settings": dict(
							mode = 'markers',
							marker = dict(
								size = 5,
								color = 'black',
								opacity = 0.3
							),
							name = "Extracted particles"
						)
					}
				]
			),
			'MS_entrance_phase_space': DataField(
				buffer_dependance = ['x_extracted_at_MS', 'px_extracted_at_MS'],
				plot_order = [
					{
						"x": 'x_extracted_at_MS',
						"y": 'px_extracted_at_MS',
						"settings": dict(
							mode = 'markers',
							marker = dict(
								size = 5,
								color = 'black',
								opacity = 0.3
							),
							name = "Extracted particles"
						)
					}
				]
			),
			'separatrix': DataField(
				buffer_dependance = ['x_stable', 'px_stable', 'x_unstable', 'px_unstable'],
				plot_order = [
					{
						"x": 'x_unstable',
						"y": 'px_unstable',
						"settings": dict(
							mode = 'markers',
							marker = dict(
								size = 5,
								color = 'red',
							),
							name = "Unstable particle",
							showlegend = True
						)
					},
					{
						"x": 'x_stable',
						"y": 'px_stable',
						"settings": dict(
							mode = 'markers',
							marker = dict(
								size = 5,
								color = 'green',
							),
							name = "Stable particle",
							showlegend = True
						)
					}
				]
			),
			'biomed_data': DataField(
				buffer_dependance = [self.time_coord, 'IC1', 'IC2', 'IC3'],
				plot_order = [
					{
						"x": self.time_coord,
						"y": "IC3",
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "green",
							),
							name = "IC3"
						)
					},
					{
						"x": self.time_coord,
						"y": "IC2",
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "red",
							),
							name = "IC2"
						)
					},
					{
						"x": self.time_coord,
						"y": "IC1",
						"settings": dict(
							mode = "lines",
							line = dict(
								color =  "blue",
							),
							name = "IC1"
						)
					},
				]
			)
		}

		self.data_to_expect, self.data_buffer = [], {}
		self.callbacks = []

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

			# storing the callbacks list separately
			if self.data_fields[data_key].callback is not None:
				self.callbacks.append(self.data_fields[data_key].callback)
		
	def calculate_loss_inside_septum(self, append_to_buffer = True): 		
		x = np.array(self.data_buffer['x_extracted_at_ES'].recent_data)
		px = np.array(self.data_buffer['px_extracted_at_ES'].recent_data)

		threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
		lost_inside_septum_mask = x > threshold
		
#		if any(lost_inside_septum_mask):
#			print(f"Entered ES septum")
#			print(f"x = {x}")
#			print(f"px = {px}")
#			print(f"Threshold based on px: {threshold}")
#			print(f"Lost mask {lost_inside_septum_mask}")
#			print()

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

	def calcualte_spill_accumulated(self):
		
		print("inside of the puffer")

		extracted = len(self.data_buffer['x_extracted_at_ES'].recent_data)
		print(extracted)
		lost_inside = sum(self.calculate_loss_inside_septum(append_to_buffer = False))

		spill_prev = self.data_buffer['spill_accumulated'].data[-1] if self.data_buffer['spill_accumulated'].data else 0

		self.data_buffer['spill_accumulated'].append(extracted - lost_inside + spill_prev)
		
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
	
	def calculate_spill_mixed_integrated(self):
		
		window_length = 100 # in turns
		
		lost_inside = self.calculate_loss_inside_septum(append_to_buffer = False)

		# separating Carbon from Helium
		ion = np.array(self.data_buffer['ion'].recent_data)
		is_C = ion == "carbon"
		is_He = ion == "helium"

		extracted_C = is_C & ~lost_inside
		extracted_He = is_He & ~lost_inside

		spill_C_prev = self.data_buffer['_spill_C_accumulated'].data[-1] if self.data_buffer['_spill_C_accumulated'].data else 0
		spill_He_prev = self.data_buffer['_spill_He_accumulated'].data[-1] if self.data_buffer['_spill_He_accumulated'].data else 0

		self.data_buffer['_spill_C_accumulated'].append(sum(extracted_C) + spill_C_prev)
		self.data_buffer['_spill_He_accumulated'].append(sum(extracted_He) + spill_He_prev)

		start, end = 0, len(self.data_buffer['_spill_C_accumulated'].data) - 1

		if end < window_length - 1:
			start = 0
		else:
			start = end - (window_length - 1)

		self.data_buffer['spill_C_integrated'].append(self.data_buffer['_spill_C_accumulated'].data[-1] - self.data_buffer['_spill_C_accumulated'].data[start])
		self.data_buffer['spill_He_integrated'].append(self.data_buffer['_spill_He_accumulated'].data[-1] - self.data_buffer['_spill_He_accumulated'].data[start])

	def calculate_spill_mixed_accumulated(self):
		lost_inside = self.calculate_loss_inside_septum(append_to_buffer = False)

		# separating Carbon from Helium
		ion = np.array(self.data_buffer['ion'].recent_data)
		is_C = ion == "carbon"
		is_He = ion == "helium"

		extracted_C = is_C & ~lost_inside
		extracted_He = is_He & ~lost_inside

		spill_C_prev = self.data_buffer['spill_C_accumulated'].data[-1] if self.data_buffer['spill_C_accumulated'].data else 0
		spill_He_prev = self.data_buffer['spill_He_accumulated'].data[-1] if self.data_buffer['spill_He_accumulated'].data else 0

		self.data_buffer['spill_C_accumulated'].append(sum(extracted_C) + spill_C_prev)
		self.data_buffer['spill_He_accumulated'].append(sum(extracted_He) + spill_He_prev)

	def calculate_total_accumulated_loss_at_septum(self):
		
		lost_inside_before = self.data_buffer['ES_septum_anode_loss_inside_accumulated'].data[-1] if self.data_buffer['ES_septum_anode_loss_inside_accumulated'].data else 0
		lost_outside_before = self.data_buffer['ES_septum_anode_loss_outside_accumulated'].data[-1] if self.data_buffer['ES_septum_anode_loss_outside_accumulated'].data else 0

		lost_inside_at_last_turn = sum(self.calculate_loss_inside_septum(append_to_buffer = False))
		lost_outside_at_last_turn = self.data_buffer['ES_septum_anode_loss_outside'].recent_data[0]

		self.data_buffer['ES_septum_anode_loss_outside_accumulated'].append(lost_outside_before + lost_outside_at_last_turn)
		self.data_buffer['ES_septum_anode_loss_inside_accumulated'].append(lost_inside_before + lost_inside_at_last_turn)

		self.data_buffer['ES_septum_anode_loss_total_accumulated'].append(
			self.data_buffer['ES_septum_anode_loss_outside_accumulated'].data[-1] + self.data_buffer['ES_septum_anode_loss_inside_accumulated'].data[-1]
		)

	def calculate_total_accumulated_loss_at_septum_mixed(self):

		lost_inside_before_C = self.data_buffer['ES_septum_anode_loss_inside_C_accumulated'].data[-1] if self.data_buffer['ES_septum_anode_loss_inside_C_accumulated'].data else 0
		lost_outside_before_C = self.data_buffer['ES_septum_anode_loss_outside_C_accumulated'].data[-1] if self.data_buffer['ES_septum_anode_loss_outside_C_accumulated'].data else 0

		lost_inside_before_He = self.data_buffer['ES_septum_anode_loss_inside_He_accumulated'].data[-1] if self.data_buffer['ES_septum_anode_loss_inside_He_accumulated'].data else 0
		lost_outside_before_He = self.data_buffer['ES_septum_anode_loss_outside_He_accumulated'].data[-1] if self.data_buffer['ES_septum_anode_loss_outside_He_accumulated'].data else 0

		lost_outside_at_last_turn_C = self.data_buffer['ES_septum_anode_loss_outside_C'].recent_data[0]
		lost_outside_at_last_turn_He = self.data_buffer['ES_septum_anode_loss_outside_He'].recent_data[0]

		lost_inside_at_last_turn = self.calculate_loss_inside_septum(append_to_buffer = False)

		ion = np.array(self.data_buffer['ion'].recent_data)
		is_C = ion == "carbon"
		is_He = ion == "helium"

		lost_inside_at_last_turn_C = is_C & lost_inside_at_last_turn
		lost_inside_at_last_turn_He = is_He & lost_inside_at_last_turn

		

		# lost inside
		self.data_buffer['ES_septum_anode_loss_inside_C_accumulated'].append(lost_inside_before_C + int(sum(lost_inside_at_last_turn_C)))
		self.data_buffer['ES_septum_anode_loss_inside_He_accumulated'].append(lost_inside_before_He + int(sum(lost_inside_at_last_turn_He)))

		# lost outside
		self.data_buffer['ES_septum_anode_loss_outside_C_accumulated'].append(lost_outside_before_C + lost_outside_at_last_turn_C)
		self.data_buffer['ES_septum_anode_loss_outside_He_accumulated'].append(lost_outside_before_He + lost_outside_at_last_turn_He)

		# total
		self.data_buffer['ES_septum_anode_loss_total_C_accumulated'].append(
			self.data_buffer['ES_septum_anode_loss_outside_C_accumulated'].data[-1] + self.data_buffer['ES_septum_anode_loss_inside_C_accumulated'].data[-1]
		)

		self.data_buffer['ES_septum_anode_loss_total_He_accumulated'].append(
			self.data_buffer['ES_septum_anode_loss_outside_He_accumulated'].data[-1] + self.data_buffer['ES_septum_anode_loss_inside_He_accumulated'].data[-1]
		)

		self.data_buffer['ES_septum_anode_loss_total_accumulated'].append(
			self.data_buffer['ES_septum_anode_loss_total_C_accumulated'].data[-1] + self.data_buffer['ES_septum_anode_loss_total_He_accumulated'].data[-1]
		)

#		print(f"Total accumulated losses")
#		print(self.data_buffer['ES_septum_anode_loss_total_accumulated'])


	def calculate_spill_mixed_diff_accumulated(self):
		lost_inside = self.calculate_loss_inside_septum(append_to_buffer = False)

		# separating Carbon from Helium
		ion = np.array(self.data_buffer['ion'].recent_data)
		is_C = ion == "carbon"
		is_He = ion == "helium"

		extracted_C = is_C & ~lost_inside
		extracted_He = is_He & ~lost_inside

		if not self.data_buffer['_C_accumulated'].data:
			self.data_buffer['_C_accumulated'].append(0)
		
		if not self.data_buffer['_He_accumulated'].data:
			self.data_buffer['_He_accumulated'].append(0)

		C_accumulated = self.data_buffer['_C_accumulated'].data[0] + sum(extracted_C)
		He_accumulated = self.data_buffer['_He_accumulated'].data[0] + sum(extracted_He)

#		print(f"C_accumulated = {C_accumulated}")
#		print(f"He_accumulated = {He_accumulated}")
#		print(f"sum(extracted_C) = {sum(extracted_C)}")
#		print(f"sum(extracted_He) = {sum(extracted_He)}")
#		print()

		if C_accumulated != 0:
			self.data_buffer['He_C_difference_accumulated'].append(He_accumulated / C_accumulated)
		else:
			self.data_buffer['He_C_difference_accumulated'].append(0)

		self.data_buffer['_C_accumulated'].data[0] = C_accumulated
		self.data_buffer['_He_accumulated'].data[0] = He_accumulated


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
				fig.update_layout(
					title = 'Es losses on the anode',
					xaxis_title = self.time_coord,
					yaxis_title = 'Lost [a.u.]',
					width = 1800,
					height = 400,
				)

			case 'ES_septum_anode_losses_accumulated':
				fig.update_layout(
					title = 'Accumulated losses on the anode',
					xaxis_title = self.time_coord,
					yaxis_title = 'Lost [a.u.]',
					width = 1500,
					height = 700,
				)

			case 'ES_septum_anode_losses_mixed_accumulated':
				fig.update_layout(
					title = 'Accumulated losses on the anode',
					xaxis_title = self.time_coord,
					yaxis_title = 'Lost [a.u.]',
					width = 1500,
					height = 700,
				)
			
			case 'ES_septum_anode_losses_inside':			
				fig.update_layout(
					title = 'Es losses on the inside of anode',
					xaxis_title = self.time_coord,
					yaxis_title = 'Lost particles [a.u.]',
					width = 1800,
					height = 400,
				)
			
			case 'ES_septum_anode_losses_outside':			
				fig.update_layout(
					title = 'Es losses on the outside of anode',
					xaxis_title = self.time_coord,
					yaxis_title = 'Lost particles [a.u.]',
					width = 1800,
					height = 400,
				)
		
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
			
			case 'spill_accumulated':
				if self.time_coord == 'time':
					fig.update_xaxes(
						type = "date",
						tickformat = "%H:%M:%S",
						tickangle = 0,
						showgrid = True,
					)

				fig.update_layout(
					title = 'Spill accumulated',
					xaxis_title = self.time_coord,
					yaxis_title = 'Spill [a.u.]',
					width = 1200,
					height = 700,
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
					title = 'Spill, mixed beam',
					xaxis_title = self.time_coord,
					yaxis_title = 'Spill',
					width = 2250,
					height = 900,
				)

			case 'spill_mixed_accumulated':

				if self.time_coord == 'time':
					fig.update_xaxes(
						type = "date",
						tickformat = "%H:%M:%S",
						tickangle = 0,
						showgrid = True,
					)
				
				fig.update_layout(
					title = 'Accumulated spill, mixed beam',
					xaxis_title = self.time_coord,
					yaxis_title = 'Spill',
					width = 1400,
					height = 700,
				)

			case 'spill_mixed_integrated':
				if self.time_coord == 'time':
					fig.update_xaxes(
						type = "date",
						tickformat = "%H:%M:%S",
						tickangle = 0,
						showgrid = True,
					)
				
				fig.update_layout(
					title = 'Integrated spill, mixed beam',
					xaxis_title = self.time_coord,
					yaxis_title = 'Spill',
					width = 1400,
					height = 700,
				)
			
			case 'spill_mixed_diff_accumulated':
				if self.time_coord == 'time':
					fig.update_xaxes(
						type = "date",
						tickformat = "%H:%M:%S",
						tickangle = 0,
						showgrid = True,
					)
				
				fig.update_layout(
					title = 'Extracted He / C, mixed beam',
					xaxis_title = self.time_coord,
					yaxis_title = 'Spill',
					width = 1200,
					height = 700,
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
				# Anode
				fig.add_shape(
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
				fig.add_shape(
					type = 'path',
					path = 'M -0.073 -0.0085 L -0.073 -0.005 L -0.083 -0.005 L -0.083 -0.0085 Z',
					fillcolor = 'rgba(0, 0, 255, 0.3)',
					line = dict(color = 'rgba(0, 0, 0, 0)'),
					name = "Cathode",
				)

				# limits on ont being lost inside of the septum
				px_loss_limit = np.linspace(-7.4e-3, -5.0e-3, 100).tolist()
				x_loss_limit = list(map(lambda px: -0.055 - (px + 7.4e-3)**2 / (2 * 1.7857e-3), px_loss_limit))

				path = f'M {x_loss_limit[0]},{px_loss_limit[0]} ' + ' '.join(
					f'L {x},{y}' for x, y in zip(x_loss_limit[1:], px_loss_limit[1:])
				)

				fig.add_shape(
					type = 'path',
					path = path,
					line = dict(
						color = 'red',
						dash = 'dash',
						width = 2,
						),
					name = "Lost inside on the wires limit",
					showlegend = True
				)


				fig.update_layout(
					title = 'Phase space at ES entrance',
					width = 800,
					height = 700,
					xaxis_title = 'x [m]',
					yaxis_title = 'px [rad]',
					showlegend = True
				)
			
			case 'MS_entrance_phase_space':
				fig.add_shape(
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

				fig.add_shape(
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

				fig.add_shape(
					x0 = 0.069, y0 = 8e-3,
					mode = 'markers',
					marker = dict(
						symbol = 'x',
						size = 12,
						color = 'red',
					),
					name = 'Septum centeer orbit',
					showlegend = True
				)

				fig.update_layout(
					title = 'Phase space at MS entrance',
					width = 800,
					height = 700,
					xaxis_title = 'x [m]',
					yaxis_title = 'px [rad]',
					showlegend = True
				)

			case 'separatrix':
				fig.add_shape(
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

				fig.update_layout(
					title = 'Separatrix',
					width = 1000,
					height = 800,
					xaxis_title = 'x [m]',
					yaxis_title = 'px [rad]',
					showlegend = True
				)

		return fig

	def run_dash_server(self):
		"""
		Start a dash server
		"""

		self.app = dash.Dash("Slow extraction w/ xsuite", title = "Extraction dashboard")
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
			if key in {'intensity', 'ES_septum_anode_losses', 'ES_septum_anode_losses_accumulated', 'ES_septum_anode_losses_mixed_accumulated', 'ES_septum_anode_losses_inside', 'ES_septum_anode_losses_outside', 'spill', 'spill_mixed', 'spill_accumulated', 'spill_mixed_accumulated', 'spill_mixed_integrated', 'spill_mixed_diff_accumulated'}:
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
			dcc.Interval(id = 'refresh', interval = 200, n_intervals = 0)
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
			
			if mode == 'file_biomed':
				data_dir = Path("data_storage")

				files = sorted(data_dir.glob("*.parquet"))
				return [{"label": f.name, "value": str(f)} for f in files]

			if mode == 'file':
				data_dir = Path(".")

				files = sorted(data_dir.glob("*.pkl"))
				print(files)
				return [{"label": f.name, "value": str(f)} for f in files]

		@self.app.callback(
			Output("cycle-selector", "options"),
			Output("cycle-selector", "value"),
			Output("load-status", "children"),
			Input("load-file-btn", "n_clicks"),
			State("file-selector", "value"),
			State("mode-switch", "value"),
			prevent_initial_call = True
		)
		def load_file_and_populate_cycles(n_clicks, filepath, mode):
			if not filepath or not Path(filepath).is_file():
				return [], None, "No file selected or not found."
			
			if mode == 'file_biomed':
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

			elif mode == 'file':
				try:
					with open(filepath, 'rb') as fid:
						self.read_from_file = xt.Particles.from_dict(pk.load(fid))
				except Exception as e:
					return [], None, f"Read error: {e}"
				
#				print(self.read_from_file)
				return [0], 0, f"Loaded particles data."
			
			else:
				return [], None, "Unknown error"

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
					# We read a file which is `xt.Particles` object, so we have to extract the 
					# data in the proper format for the buffers
					
					print(self.data_to_monitor)
					print(self.data_to_expect)
					# The data put in the buffers should depend on the data fields requested
					with open(filepath, 'rb') as fid:
						self.read_from_file = xt.Particles.from_dict(pk.load(fid))
					self.read_from_file.sort(by = 'at_turn', interleave_lost_particles = True)

					self._clear_buffer()		

					max_turns = max(self.read_from_file.at_turn)
					turns_list = list(range(max_turns + 1))
					
					# basic masks
					lost_mask = self.read_from_file.state == 0
					at_start = abs(self.read_from_file.s) < 1e-7
					extracted_at_ES = lost_mask & at_start

					# Nparticles buffer
					if 'Nparticles' in self.data_to_expect:
						lost_particles = self.read_from_file.filter(lost_mask)
						print("Particles lost in the tracking")
						print(lost_particles.get_table())
						particles_alive_for_turns = self.read_from_file._capacity - np.searchsorted(lost_particles.at_turn, turns_list, side = "left")

					# phase space buffer
					if 'x_extracted_at_ES' in self.data_to_expect or 'px_extracted_at_ES' in self.data_to_expect:
						lost_particles_at_ES_septum = self.read_from_file.filter(extracted_at_ES)

					# losses on the septum wires
					if 'ES_septum_anode_loss_outside' in self.data_to_expect:
						at_septum_end = self.read_from_file.s == 1.5
						lost_at_septum_end = lost_mask & at_septum_end
						lost_particles_at_septum_end = self.read_from_file.filter(lost_at_septum_end)						
						
						print("Particles lost on the outside of a septum")
						print(lost_particles_at_septum_end.get_table())

						number_lost_particles_at_septum_end = np.bincount(
							lost_particles_at_septum_end.at_turn,
							minlength = max_turns
						)

					# since I do not use callbacks from listener, I have to populate some buffers manually here
					if 'ES_septum_anode_losses_accumulated' in self.data_to_monitor:
						number_lost_particles_at_septum_end_accumulated = np.cumsum(number_lost_particles_at_septum_end)

					if any(key in self.data_to_monitor for key in (
						'ES_septum_anode_losses_inside', 
						'ES_septum_anode_losses', 
						'ES_septum_anode_losses_accumulated', 
						'spill', 
						'spill_accumulated'
						)):
						x = lost_particles_at_ES_septum.x
						px = lost_particles_at_ES_septum.px

						threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
						lost_inside_septum = x > threshold
						lost_particles_inside_of_septum = lost_particles_at_ES_septum.filter(lost_inside_septum)
						
						print("Particles lost on the wires inside of a septum")
						print(lost_particles_inside_of_septum.get_table())

						number_lost_particles_inside_of_septum = np.bincount(
							lost_particles_inside_of_septum.at_turn,
							minlength = max_turns
						)
					if 'ES_septum_anode_losses' in self.data_to_monitor:
						septum_losses = number_lost_particles_inside_of_septum + number_lost_particles_at_septum_end

					if 'ES_septum_anode_losses_accumulated' in self.data_to_monitor:
						number_lost_particles_inside_of_septum_accumulated = np.cumsum(number_lost_particles_inside_of_septum)
						septum_losses_accumulated = number_lost_particles_inside_of_septum_accumulated + number_lost_particles_at_septum_end_accumulated

					if any(key in self.data_to_monitor for key in ('spill', 'spill_accumulated')):
						entered_septum = np.bincount(
							lost_particles_at_ES_septum.at_turn,
							minlength = max_turns
						)
						extracted_at_ES_at_turn = entered_septum - number_lost_particles_inside_of_septum

					if 'spill_accumulated' in self.data_to_monitor:
						extracted_at_ES_at_turn_acc = np.cumsum(extracted_at_ES_at_turn)

					with self._buflock:
						# data buffers
						if 'turn' in self.data_to_expect:
							self.data_buffer['turn'].extend(turns_list)

						if 'Nparticles' in self.data_to_expect:
							self.data_buffer['Nparticles'].extend(particles_alive_for_turns)

						if 'x_extracted_at_ES' in self.data_to_expect:
							self.data_buffer['x_extracted_at_ES'].extend(lost_particles_at_ES_septum.x)

						if 'px_extracted_at_ES' in self.data_to_expect:
							self.data_buffer['px_extracted_at_ES'].extend(lost_particles_at_ES_septum.px)
						
						if 'ES_septum_anode_loss_outside' in self.data_to_expect:
							self.data_buffer['ES_septum_anode_loss_outside'].extend(number_lost_particles_at_septum_end)
						
						# data fields
						if any(key in self.data_to_monitor for key in ('ES_septum_anode_losses_inside', 'ES_septum_anode_losses')):
							self.data_buffer['ES_septum_anode_loss_inside'].extend(number_lost_particles_inside_of_septum)

						if 'ES_septum_anode_losses_accumulated' in self.data_to_monitor:
							self.data_buffer['ES_septum_anode_loss_outside_accumulated'].extend(number_lost_particles_at_septum_end_accumulated)
							self.data_buffer['ES_septum_anode_loss_inside_accumulated'].extend(number_lost_particles_inside_of_septum_accumulated)
							self.data_buffer['ES_septum_anode_loss_total_accumulated'].extend(septum_losses_accumulated)

						if 'ES_septum_anode_losses' in self.data_to_monitor:
							self.data_buffer['ES_septum_anode_loss_total'].extend(septum_losses)

						if 'spill' in self.data_to_monitor:
							self.data_buffer['spill'].extend(extracted_at_ES_at_turn)
						
						if 'spill_accumulated' in self.data_to_monitor:
							self.data_buffer['spill_accumulated'].extend(extracted_at_ES_at_turn_acc)


				elif mode == "file_biomed":
					single_cycle = self.read_from_file[self.read_from_file['cycle_id'] == cycle_id]
					print(single_cycle)

					self._clear_buffer()

					with self._buflock:
						
						self.data_buffer['IC1'].extend(list(single_cycle['Y[0]'].values))
						self.data_buffer['IC2'].extend(list(single_cycle['Y[1]'].values))
						self.data_buffer['IC3'].extend(list(single_cycle['Y[2]'].values))

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