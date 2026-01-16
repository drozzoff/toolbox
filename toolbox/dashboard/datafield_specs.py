from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
from functools import partial
import numpy as np

from toolbox.dashboard.callbacks import *
from toolbox.dashboard.plot_layouts import *


@dataclass
class DataField:
	"""Simple class to store the dependances and state of the data field in the dashboard"""
	buffer_dependance: list[str] = field(default_factory = list)
	output_buffers: list[str] | None = None
	callback: Optional[callable] = None
	callback_level: int | None = None
	state: bool = False
	plot_from: list[str] | None = None # list of the buffers that this data field is dependent upon for plotting
	plot_layout: callable | None = None

	buffer_pointer: int = 0

	plot_order: list[dict] | None = None # Description of the order traces are added to the plot

	category: str | None = None
	

def make_datafields(dashboard: ExtractionDashboard):
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
						showlegend = True
					)
				}
			],
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
						showlegend = True
					)
				}
			],
			plot_layout = ES_outside_losses_layout,
			category = "Turn By Turn"
		),
		'ES_septum_losses:inside': DataField(
			buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn'],
			output_buffers = ['ES_septum_losses:inside'],
			callback = partial(ES_inside_losses_callback, dashboard),
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
						showlegend = True
					)
				}
			],
			plot_layout = ES_inside_losses_layout,
			category = "Turn By Turn"
		),
		'ES_septum_losses': DataField(
			buffer_dependance = ['turn', 'ES_septum_losses:inside', 'lost_on_septum_wires'],
			output_buffers = ['ES_septum_losses'],
			callback = partial(ES_losses_callback, dashboard),
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
			plot_layout = ES_losses_layout,
			category = "Turn By Turn"
		),
		'spill': DataField(
			buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn'], 
			output_buffers = ['spill'],
			callback = partial(spill_callback, dashboard),
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
		'biomed_data': DataField(
			buffer_dependance = ['time', 'IC1', 'IC2', 'IC3'],
			plot_order = [
				{
					"x": 'time',
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
					"x": 'time',
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
					"x": 'time',
					"y": "IC1",
					"settings": dict(
						mode = "lines",
						line = dict(
							color =  "blue",
						),
						name = "IC1"
					)
				},
			],
			plot_layout = biomed_data_layout,
			category = "Biomed"
		),
	}

def process_particles_file(dashboard: ExtractionDashboard, particles: xt.Particles) -> dict:
	"""
	Maps the data needed extracted from the file according to `dashboard.data_to_expect`
	"""
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
			lost_particles_at_ES_septum = None
	
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
			lost_at_septum_end = lost_mask & at_septum_end

			if np.sum(lost_at_septum_end) > 0:
				lost_particles_at_septum_end = particles.filter(lost_at_septum_end)						

				number_lost_particles_at_septum_end = np.bincount(
					lost_particles_at_septum_end.at_turn,
					minlength = max_turns
				)
			else: number_lost_particles_at_septum_end = np.zeros(max_turns)

			data_mapping[key] = number_lost_particles_at_septum_end
	
	return data_mapping

"""
	ASSORTED DATA

		'ES_septum_anode_losses_mixed_accumulated': DataField(
			buffer_dependance = [
				'turn', 
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
			callback = dashboard.calculate_total_accumulated_loss_at_septum_mixed,
			plot_from = [
				'turn', 
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
					"x": 'turn',
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
					"x": 'turn',
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
					"x": 'turn',
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
					"x": 'turn',
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
					"x": 'turn',
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
					"x": 'turn',
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
					"x": 'turn',
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
		'spill_mixed': DataField(
			buffer_dependance = ['turn', 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
			create_new_buffer = ['spill_C', 'spill_He'],
			callback = dashboard.calculate_spill_mixed,
			plot_from = ['turn', 'spill_C', 'spill_He'],
			plot_order = [
				{
					"x": 'turn',
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
					"x": 'turn',
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
			buffer_dependance = ['turn', 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
			create_new_buffer = ['_spill_C_accumulated', '_spill_He_accumulated', 'spill_C_integrated', 'spill_He_integrated'],
			callback = dashboard.calculate_spill_mixed_integrated,
			plot_from = ['turn', 'spill_C_integrated', 'spill_He_integrated'],
			plot_order = [
				{
					"x": 'turn',
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
					"x": 'turn',
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
			buffer_dependance = ['turn', 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
			create_new_buffer = ['spill_C_accumulated', 'spill_He_accumulated'],
			callback = dashboard.calculate_spill_mixed_accumulated,
			plot_from = ['turn', 'spill_C_accumulated', 'spill_He_accumulated'],
			plot_order = [
				{
					"x": 'turn',
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
					"x": 'turn',
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
			buffer_dependance = ['turn', 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
			create_new_buffer = ['He_C_difference_accumulated', '_C_accumulated', '_He_accumulated'],
			callback = dashboard.calculate_spill_mixed_diff_accumulated,
			plot_from = ['turn', 'He_C_difference_accumulated'],
			plot_order = [
				{
					"x": 'turn',
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
		)
"""