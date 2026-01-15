from typing import Optional
from dataclasses import dataclass, field
from functools import partial
import numpy as np

@dataclass
class DataField:
	"""Simple class to store the dependances and state of the data field in the dashboard"""
	buffer_dependance: list[str] = field(default_factory = list)
	output_buffers: list[str] | None = None
	callback: Optional[callable] = None
	state: bool = False

	plot_from: list[str] | None = None # list of the buffers that this data field is dependent upon for plotting

	buffer_pointer: int = 0

	plot_order: list[dict] | None = None # Description of the order traces are added to the plot

def ES_inside_losses_callback(dashboard):
	ref_turn = dashboard.data_buffer['extracted_at_ES:x'].recent_data
	x = np.array(dashboard.data_buffer['extracted_at_ES:x'].recent_data)
	px = np.array(dashboard.data_buffer['extracted_at_ES:px'].recent_data)
	at_turn = np.array(dashboard.data_buffer['extracted_at_ES:at_turn'].recent_data)

	threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
	lost_inside_septum_at_turn = at_turn[x > threshold]

	losses_at_turn = np.bincount(
		lost_inside_septum_at_turn,
		minlength = max_turns
	)
	dashboard.data_buffer['ES_septum_losses:inside'].append(np.sum(lost_inside_septum_mask))

def ES_losses_callback(dashboard):
	
	lost_inside = np.array(dashboard.data_buffer['ES_septum_losses:inside'].recent_data)
	lost_ooutside

	dashboard.data_buffer['ES_septum_losses:inside'].append(np.sum(lost_inside_septum_mask))

def make_datafields(dashboard):
	time_coord = dashboard.time_coord
	return {
		'intensity': DataField(
			buffer_dependance = [time_coord, 'Nparticles'],
			plot_from = [time_coord, 'Mparticles'],
			plot_order = [
				{
					"x": time_coord,
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
		'ES_septum_losses:outside': DataField(
			buffer_dependance = [time_coord, 'ES_septum_losses:outside'],
			plot_from = [time_coord, 'ES_septum_losses:outside'],
			plot_order = [
				{
					"x": time_coord,
					"y": 'ES_septum_losses:outside',
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
		'ES_septum_losses:inside': DataField(
			buffer_dependance = [time_coord, 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn'],
			output_buffers = ['ES_septum_losses:inside'],
			callback = partial(ES_inside_losses_callback, dashboard),
			plot_from = [time_coord, 'ES_septum_losses:inside'],
			plot_order = [
				{
					"x": time_coord,
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
			]
		),
		'ES_septum_losses': DataField(
			buffer_dependance = [time_coord, 'ES_septum_losses:inside', 'ES_septum_losses:outside'],
			output_buffers = ['ES_septum_losses'],
			callback = ES_losses_callback(dashboard),
			plot_from = [time_coord, 'ES_septum_losses:inside', 'ES_septum_losses:outside', 'ES_septum_losses'],
			plot_order = [
				{
					"x": time_coord,
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
					"x": time_coord,
					"y": 'ES_septum_losses:outside',
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
					"x": time_coord,
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

			]
		),
		'spill': DataField(
			buffer_dependance = [time_coord, 'ES_septum_losses:inside', 'ES_septum_losses:outside'], 
			output_buffers = ['spill'],
			callback = dashboard.calculate_spill,
			plot_from = [time_coord, 'spill'],
			plot_order = [
				{
					"x": time_coord,
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
		'ES_septum_anode_losses_accumulated': DataField(
			buffer_dependance = [time_coord, 'ES_septum_anode_loss_outside', 'x_extracted_at_ES', 'px_extracted_at_ES'],
			create_new_buffer = ['ES_septum_anode_loss_outside_accumulated', 'ES_septum_anode_loss_inside_accumulated', 'ES_septum_anode_loss_total_accumulated'],
			callback = dashboard.calculate_total_accumulated_loss_at_septum,
			plot_from = [time_coord, 'ES_septum_anode_loss_outside_accumulated', 'ES_septum_anode_loss_inside_accumulated', 'ES_septum_anode_loss_total_accumulated'],
			plot_order = [
				{
					"x": time_coord,
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
					"x": time_coord,
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
					"x": time_coord,
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
				time_coord, 
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
				time_coord, 
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
					"x": time_coord,
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
					"x": time_coord,
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
					"x": time_coord,
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
					"x": time_coord,
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
					"x": time_coord,
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
					"x": time_coord,
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
					"x": time_coord,
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
		'spill_accumulated': DataField(
			buffer_dependance = [time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES'], 
			create_new_buffer = ['spill_accumulated'],
			callback = dashboard.calcualte_spill_accumulated,
			plot_from = [time_coord, 'spill_accumulated'],
			plot_order = [
				{
					"x": time_coord,
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
			buffer_dependance = [time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
			create_new_buffer = ['spill_C', 'spill_He'],
			callback = dashboard.calculate_spill_mixed,
			plot_from = [time_coord, 'spill_C', 'spill_He'],
			plot_order = [
				{
					"x": time_coord,
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
					"x": time_coord,
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
			buffer_dependance = [time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
			create_new_buffer = ['_spill_C_accumulated', '_spill_He_accumulated', 'spill_C_integrated', 'spill_He_integrated'],
			callback = dashboard.calculate_spill_mixed_integrated,
			plot_from = [time_coord, 'spill_C_integrated', 'spill_He_integrated'],
			plot_order = [
				{
					"x": time_coord,
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
					"x": time_coord,
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
			buffer_dependance = [time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
			create_new_buffer = ['spill_C_accumulated', 'spill_He_accumulated'],
			callback = dashboard.calculate_spill_mixed_accumulated,
			plot_from = [time_coord, 'spill_C_accumulated', 'spill_He_accumulated'],
			plot_order = [
				{
					"x": time_coord,
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
					"x": time_coord,
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
			buffer_dependance = [time_coord, 'x_extracted_at_ES', 'px_extracted_at_ES', 'ion'], 
			create_new_buffer = ['He_C_difference_accumulated', '_C_accumulated', '_He_accumulated'],
			callback = dashboard.calculate_spill_mixed_diff_accumulated,
			plot_from = [time_coord, 'He_C_difference_accumulated'],
			plot_order = [
				{
					"x": time_coord,
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
			buffer_dependance = [time_coord, 'IC1', 'IC2', 'IC3'],
			plot_order = [
				{
					"x": time_coord,
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
					"x": time_coord,
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
					"x": time_coord,
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