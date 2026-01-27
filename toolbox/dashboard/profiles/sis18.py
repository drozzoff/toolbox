from __future__ import annotations
from functools import partial
import numpy as np
import xtrack as xt
import pickle as pk
from toolbox.dashboard.profiles.datafield import DataField


class SIS18Profile:
	def __init__(self, start_count_at_turn: int = 0):
		self.start_count_at_turn = start_count_at_turn

	name = "SIS18 KO"
	def make_datafields(self, dashboard: ExtractionDashboard):
		return {
			'intensity': DataField(
				buffer_dependance = ['turn', 'Nparticles'],
				plot_order = [
					{
						"x": 'turn',
						"y": 'Nparticles',
						"settings": dict(
							mode = "lines",
							line = {"color": "blue"},
							name = "Intensity in the ring",
							showlegend = False
						)
					}
				],
				bin = dict(enabled = True, x = "middle", y = "mean"),
				plot_layout = intensity_layout,
				category = "Turn By Turn"
			),
			'ES_septum_losses:outside': DataField(
				buffer_dependance = ['turn', 'lost_on_septum_wires'],
				plot_order = [
					{
						"x": 'turn',
						"y": 'lost_on_septum_wires',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'blue',
								'width': 2,
							},
							name = "Lost outside of the septum",
							showlegend = False
						)
					}
				],
				bin = dict(enabled = True, x = "middle", y = "sum"),
				plot_layout = ES_outside_losses_layout,
				category = "Turn By Turn"
			),
			'ES_septum_losses:inside': DataField(
				buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn'],
				output_buffers = ['ES_septum_losses:inside'],
				callback = partial(ES_inside_losses_callback, dashboard, start_count_at_turn = self.start_count_at_turn),
				callback_level = 0,
				plot_from = ['turn', 'ES_septum_losses:inside'],
				plot_order = [
					{
						"x": 'turn',
						"y": 'ES_septum_losses:inside',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'blue',
								'width': 2,
							},
							name = "Lost inside of the septum",
							showlegend = False
						)
					}
				],
				bin = dict(enabled = True, x = "middle", y = "sum"),
				plot_layout = ES_inside_losses_layout,
				category = "Turn By Turn"
			),
			'ES_septum_losses': DataField(
				buffer_dependance = ['turn', 'ES_septum_losses:inside', 'lost_on_septum_wires'],
				output_buffers = ['ES_septum_losses'],
				callback = partial(ES_losses_callback, dashboard, start_count_at_turn = self.start_count_at_turn),
				callback_level = 0,
				plot_from = ['turn', 'ES_septum_losses:inside', 'lost_on_septum_wires', 'ES_septum_losses'],
				plot_order = [
					{
						"x": 'turn',
						"y": 'ES_septum_losses',
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
						"x": 'turn',
						"y": 'lost_on_septum_wires',
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
						"x": 'turn',
						"y": 'ES_septum_losses:inside',
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

				],
				bin = dict(enabled = True, x = "middle", y = "sum"),
				plot_layout = ES_losses_layout,
				category = "Turn By Turn"
			),
			'spill': DataField(
				buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn'], 
				output_buffers = ['spill'],
				callback = partial(spill_callback, dashboard, start_count_at_turn = self.start_count_at_turn),
				callback_level = 0,
				plot_from = ['turn', 'spill'],
				plot_order = [
					{
						"x": 'turn',
						"y": 'spill',
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "blue"
							),
							name = "Spill"
						)
					},
				],
				bin = dict(enabled = True, x = "middle", y = "sum"),
				plot_layout = spill_layout,
				category = "Turn By Turn"
			),
			'spill:accumulated': DataField(
				buffer_dependance = ['turn', 'spill'], 
				output_buffers = ['spill:accumulated'],
				callback = partial(accumulated_spill_callback, dashboard),
				callback_level = 1,
				plot_from = ['turn', 'spill:accumulated'],
				plot_order = [
					{
						"x": 'turn',
						"y": 'spill:accumulated',
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "blue"
							),
							name = "Spill"
						)
					},
				],
				bin = dict(enabled = True, x = "last", y = "last"),
				plot_layout = accumulated_spill_layout,
				category = "Turn By Turn"
			),
			'ES_septum_losses:outside:accumulated': DataField(
				buffer_dependance = ['turn', 'lost_on_septum_wires'],
				output_buffers = ['ES_septum_losses:outside:accumulated'],
				callback = partial(accumulated_outside_ES_losses_callback, dashboard),
				callback_level = 0,
				plot_from = ['turn', 'ES_septum_losses:outside:accumulated', ],
				plot_order = [
					{
						"x": 'turn',
						"y": 'ES_septum_losses:outside:accumulated',
						"settings": dict(
							mode = 'lines',
							line = {
								'color': 'red',
								'width': 2,
							},
							name = "Lost outside",
							showlegend = True
						)
					}
				],
				bin = dict(enabled = True, x = "last", y = "last"),
				plot_layout = accumulated_ES_losses_layout,
				category = "Turn By Turn"
			),
			'ES_septum_losses:inside:accumulated': DataField(
				buffer_dependance = ['turn', 'ES_septum_losses:inside'],
				output_buffers = ['ES_septum_losses:inside:accumulated'],
				callback = partial(accumulated_inside_ES_losses_callback, dashboard),
				callback_level = 1,
				plot_from = ['turn', 'ES_septum_losses:inside:accumulated'],
				plot_order = [
					{
						"x": 'turn',
						"y": 'ES_septum_losses:inside:accumulated',
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
				],
				bin = dict(enabled = True, x = "last", y = "last"),
				plot_layout = accumulated_ES_losses_layout,
				category = "Turn By Turn"
			),
			'ES_septum_losses:accumulated': DataField(
				buffer_dependance = ['turn', 'ES_septum_losses:outside', 'ES_septum_losses:inside'],
				output_buffers = ['ES_septum_losses:accumulated', 'ES_septum_losses:outside:accumulated', 'ES_septum_losses:inside:accumulated'],
				callback = partial(accumulated_ES_losses_callback, dashboard),
				callback_level = 1,
				plot_from = ['turn', 'ES_septum_losses:inside:accumulated', 'ES_septum_losses:outside:accumulated', 'ES_septum_losses:accumulated'],
				plot_order = [
					{
						"x": 'turn',
						"y": 'ES_septum_losses:accumulated',
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
						"x": 'turn',
						"y": 'ES_septum_losses:outside:accumulated',
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
						"x": 'turn',
						"y": 'ES_septum_losses:inside:accumulated',
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
				],
				bin = dict(enabled = True, x = "last", y = "last"),
				plot_layout = accumulated_ES_losses_layout,
				category = "Turn By Turn"
			),
			'ES_entrance_phase_space': DataField(
				buffer_dependance = ['extracted_at_ES:x', 'extracted_at_ES:px'],
				plot_order = [
					{
						"x": 'extracted_at_ES:x',
						"y": 'extracted_at_ES:px',
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
				],
				plot_layout = ES_entrance_phase_space_layout,
				category = "Phase Space"
			),
		}

	def read_file(self, filename: str) -> xt.Particles:
		with open(filename, 'rb') as fid:
			particles = xt.Particles.from_dict(pk.load(fid))

		particles.sort(by = 'at_turn', interleave_lost_particles = True)

		return particles

	def process_file(
			self, 
			dashboard: ExtractionDashboard, 
			particles: xt.Particles | str, 
			start_count_at_turn: None | int = None,
			**kwargs
		)-> dict:
		"""
		Maps the data needed extracted from the file according to `dashboard.data_to_expect`
		"""
		if start_count_at_turn is None:
			start_count_at_turn = self.start_count_at_turn

		if isinstance(particles, str):
			particles = self.read_file(particles)

		max_turns = max(particles.at_turn)
		turns_list = list(range(max_turns + 1))
		
		# basic masks
		lost_mask = particles.state == 0
		at_start = np.abs(particles.s) < 1e-7

		data_mapping = {}

		if any(np.isin(['extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn'], dashboard.data_to_expect)):
			extracted_at_ES = lost_mask & at_start
			if np.sum(extracted_at_ES) > 0:
				lost_particles_at_ES_septum = particles.filter(extracted_at_ES)
			else:
				lost_particles_at_ES_septum = None # means no particles are lost
		
		data_mapping = {}

		for key in dashboard.data_to_expect:
			if key == 'turn':
				data_mapping[key] = turns_list

			if key == 'Nparticles':
				if np.sum(lost_mask) > 0:
					lost_particles = particles.filter(lost_mask)
					particles_alive_for_turns = particles._capacity - np.searchsorted(lost_particles.at_turn, turns_list, side = "left")
				else:
					particles_alive_for_turns = particles._capacity * np.ones_like(turns_list)

				data_mapping[key] = particles_alive_for_turns

			if key == 'extracted_at_ES:x':
				data_mapping[key] = lost_particles_at_ES_septum.x
			if key == 'extracted_at_ES:px':
				data_mapping[key] = lost_particles_at_ES_septum.px
			if key == 'extracted_at_ES:at_turn':
				data_mapping[key] = lost_particles_at_ES_septum.at_turn

			if key == 'lost_on_septum_wires':
				at_septum_end = particles.s == 1.5
				good_turn = particles.at_turn > start_count_at_turn
				lost_at_septum_end = lost_mask & at_septum_end
				lost_at_septum_end_good_turn = lost_at_septum_end & good_turn
				
				if np.sum(lost_at_septum_end_good_turn) > 0:
					
					lost_particles_at_septum_end = particles.filter(lost_at_septum_end_good_turn)						

					number_lost_particles_at_septum_end = np.bincount(
						lost_particles_at_septum_end.at_turn,
						minlength = max_turns
					)
				else: number_lost_particles_at_septum_end = np.zeros(max_turns)

				data_mapping[key] = number_lost_particles_at_septum_end
		
		return data_mapping

# callbacks
def ES_inside_losses_callback(dashboard: ExtractionDashboard, start_count_at_turn: int = 0):

	x = np.array(dashboard.data_buffer['extracted_at_ES:x'].recent_data)
	px = np.array(dashboard.data_buffer['extracted_at_ES:px'].recent_data)

	start_turn = dashboard.data_buffer['turn'].recent_data[0]
	end_turn = dashboard.data_buffer['turn'].recent_data[-1]

	at_turn = np.array(dashboard.data_buffer['extracted_at_ES:at_turn'].recent_data) - start_turn

	threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
	good_turns = at_turn > start_count_at_turn

	lost_inside_septum_at_turn = at_turn[(x > threshold) & good_turns]
	
	losses_at_turn = np.bincount(
		lost_inside_septum_at_turn,
		minlength = end_turn - start_turn
	)
	dashboard.data_buffer['ES_septum_losses:inside'].extend(losses_at_turn, batch_id = dashboard.current_batch_id)

def ES_losses_callback(dashboard: ExtractionDashboard, start_count_at_turn: int = 0):
	if dashboard.data_buffer['ES_septum_losses:inside'].last_batch_id != dashboard.current_batch_id:
		ES_inside_losses_callback(dashboard, start_count_at_turn)

	lost_inside = np.array(dashboard.data_buffer['ES_septum_losses:inside'].recent_data)
	lost_outside = np.array(dashboard.data_buffer['lost_on_septum_wires'].recent_data)

	dashboard.data_buffer['ES_septum_losses'].extend(lost_inside + lost_outside, batch_id = dashboard.current_batch_id)

def spill_callback(dashboard: ExtractionDashboard, start_count_at_turn: int = 0):
	x = np.array(dashboard.data_buffer['extracted_at_ES:x'].recent_data)
	px = np.array(dashboard.data_buffer['extracted_at_ES:px'].recent_data)

	start_turn = dashboard.data_buffer['turn'].recent_data[0]
	end_turn = dashboard.data_buffer['turn'].recent_data[-1]

	at_turn = np.array(dashboard.data_buffer['extracted_at_ES:at_turn'].recent_data) - start_turn

	threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
	lost_inside_septum = x > threshold
	good_turns = at_turn > start_count_at_turn
	good_particles = ~lost_inside_septum & good_turns
	survived_inside_septum_at_turn = at_turn[good_particles]

	losses_at_turn = np.bincount(
		survived_inside_septum_at_turn,
		minlength = end_turn - start_turn
	)
	dashboard.data_buffer['spill'].extend(losses_at_turn, batch_id = dashboard.current_batch_id)

def _accumulated_quantity(dashboard: ExtractionDashboard, buffer_key: str, **kwargs):
	"""
	takes `buffer_key` and pushes the data to `"{buffer_key}:accumulated"`
	"""
	acc_buffer_key = kwargs.get('new_buffer_name', f"{buffer_key}:accumulated")

	recent_value = dashboard.data_buffer[acc_buffer_key].data[-1] if dashboard.data_buffer[acc_buffer_key].data else 0

	data = np.array(dashboard.data_buffer[buffer_key].recent_data)
	extension = recent_value + np.cumsum(data)

	dashboard.data_buffer[acc_buffer_key].extend(extension, batch_id = dashboard.current_batch_id)

def accumulated_spill_callback(dashboard: ExtractionDashboard):
	if dashboard.data_buffer['spill'].last_batch_id != dashboard.current_batch_id:
		spill_callback(dashboard)

	_accumulated_quantity(dashboard, 'spill')

def accumulated_outside_ES_losses_callback(dashboard: ExtractionDashboard):
	_accumulated_quantity(dashboard, 'lost_on_septum_wires', new_buffer_name = 'ES_septum_losses:outside:accumulated')

def accumulated_inside_ES_losses_callback(dashboard: ExtractionDashboard):
	if dashboard.data_buffer['ES_septum_losses:inside'].last_batch_id != dashboard.current_batch_id:
		ES_inside_losses_callback(dashboard)

	_accumulated_quantity(dashboard, 'ES_septum_losses:inside')

def accumulated_ES_losses_callback(dashboard: ExtractionDashboard):
	if dashboard.data_buffer['ES_septum_losses:inside:accumulated'].last_batch_id != dashboard.current_batch_id:
		accumulated_inside_ES_losses_callback(dashboard)

	if dashboard.data_buffer['ES_septum_losses:outside:accumulated'].last_batch_id != dashboard.current_batch_id:
		accumulated_outside_ES_losses_callback(dashboard)

	lost_inside = np.array(dashboard.data_buffer['ES_septum_losses:inside:accumulated'].recent_data)
	lost_outside = np.array(dashboard.data_buffer['ES_septum_losses:outside:accumulated'].recent_data)

	dashboard.data_buffer['ES_septum_losses:accumulated'].extend(lost_inside + lost_outside, batch_id = dashboard.current_batch_id)

# plotting layouts
def intensity_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Intensity',
		xaxis_title = 'turn',
		yaxis_title = 'Intensity [a.u.]',
		height = 400,
	)

def ES_outside_losses_layout(fig: go.Figure):		
	fig.update_layout(
		title = 'Es losses on the outside of anode',
		xaxis_title = 'turn',
		yaxis_title = 'Lost particles [a.u.]',
		height = 400,
	)

def ES_inside_losses_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Es losses on the inside of anode',
		xaxis_title = 'turn',
		yaxis_title = 'Lost particles [a.u.]',
		height = 400,
	)

def ES_losses_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Es losses on the anode',
		xaxis_title = 'turn',
		yaxis_title = 'Lost [a.u.]',
		height = 400,
	)

def spill_layout(fig: go.Figure):
	if False:
		fig.update_xaxes(
			type = "date",
			tickformat = "%H:%M:%S",
			tickangle = 0,
			showgrid = True,
		)

	fig.update_layout(
		title = 'Spill',
		xaxis_title = 'turn',
		yaxis_title = 'Spill [a.u.]',
		height = 400,
		showlegend = False
	)

def accumulated_spill_layout(fig: go.Figure):
	if False:
		fig.update_xaxes(
			type = "date",
			tickformat = "%H:%M:%S",
			tickangle = 0,
			showgrid = True,
		)

	fig.update_layout(
		title = 'Spill accumulated',
		xaxis_title = 'turn',
		yaxis_title = 'Spill [a.u.]',
		height = 700,
		showlegend = False
	)

def accumulated_ES_losses_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Accumulated losses on the anode',
		xaxis_title = 'turn',
		yaxis_title = 'Lost [a.u.]',
		height = 700,
	)

def ES_entrance_phase_space_layout(fig: go.Figure):
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