from __future__ import annotations
from functools import partial

from toolbox.dashboard.profiles.datafield import DataField
from toolbox.dashboard.profiles.SIS18_mixed_beam.callbacks import *
from toolbox.dashboard.profiles.SIS18_mixed_beam.plot_layouts import *
import toolbox.dashboard.profiles.SIS18.datafield_specs as basic_datafields

def make_datafields(dashboard: ExtractionDashboard):
	# reading the basic 
	res = basic_datafields.make_datafields(dashboard)
	
	# and extending for mixed ions beam
	res['spill:ion1'] =  DataField(
		buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn', 'extracted_at_ES:ion'], 
		output_buffers = ['spill:ion1'],
		callback = partial(ion_spill_callback, dashboard, 1),
		plot_from = ['turn', 'spill:ion1'],
		plot_order = [
			{
				"x": 'turn',
				"y": 'spill:ion1',
				"settings": dict(
					mode = "lines",
					line = dict(
						color = "blue"
					),
					name = "Ion 1"
				)
			}
		],
		plot_layout = spill_layout,
		category = "Turn By Turn"
	)

	res['spill:ion2'] =  DataField(
		buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn', 'extracted_at_ES:ion'], 
		output_buffers = ['spill:ion2'],
		callback = partial(ion_spill_callback, dashboard, 2),
		plot_from = ['turn', 'spill:ion2'],
		plot_order = [
			{
				"x": 'turn',
				"y": 'spill:ion2',
				"settings": dict(
					mode = "lines",
					line = dict(
						color = "blue"
					),
					name = "Ion 2"
				)
			}
		],
		plot_layout = spill_layout,
		category = "Turn By Turn"
	)

	res['spill:mixed'] =  DataField(
		buffer_dependance = ['turn', 'spill:ion1', 'spill:ion2'], 
		callback = partial(mixed_spill_callback, dashboard),
		plot_from = ['turn', 'spill:ion1', 'spill:ion2'],
		plot_order = [
			{
				"x": 'turn',
				"y": 'spill:ion1',
				"settings": dict(
					mode = "lines",
					line = dict(
						color = "blue"
					),
					name = "Ion 1"
				)
			},
			{
				"x": 'turn',
				"y": 'spill:ion2',
				"settings": dict(
					mode = "lines",
					line = dict(
						color = "red"
					),
					name = "Ion 2"
				)
			}
		],
		plot_layout = spill_layout,
		category = "Turn By Turn"
	)

	res['spill:ion1:accumulated'] =  DataField(
		buffer_dependance = ['turn', 'spill:ion1'], 
		output_buffers = ['spill:ion1:accumulated'],
		callback = partial(ion_accumulated_spill_callback, dashboard, 1),
		plot_from = ['turn', 'spill:ion1'],
		plot_order = [
			{
				"x": 'turn',
				"y": 'spill:ion1',
				"settings": dict(
					mode = "lines",
					line = dict(
						color = "blue"
					),
					name = "Ion 1"
				)
			}
		],
		plot_layout = accumulated_spill_layout,
		category = "Turn By Turn"
	)

	res['spill:ion2:accumulated'] =  DataField(
		buffer_dependance = ['turn', 'spill:ion2'], 
		output_buffers = ['spill:ion2:accumulated'],
		callback = partial(ion_accumulated_spill_callback, dashboard, 2),
		plot_from = ['turn', 'spill:ion2'],
		plot_order = [
			{
				"x": 'turn',
				"y": 'spill:ion2',
				"settings": dict(
					mode = "lines",
					line = dict(
						color = "blue"
					),
					name = "Ion 2"
				)
			}
		],
		plot_layout = accumulated_spill_layout,
		category = "Turn By Turn"
	)

	res['spill:mixed:accumulated'] =  DataField(
		buffer_dependance = ['turn', 'spill:ion1:accumulated', 'spill:ion2:accumulated'], 
		callback = partial(mixed_spill_callback, dashboard),
		plot_from = ['turn', 'spill:ion1:accumulated', 'spill:ion2:accumulated'],
		plot_order = [
			{
				"x": 'turn',
				"y": 'spill:ion1:accumulated',
				"settings": dict(
					mode = "lines",
					line = dict(
						color = "blue"
					),
					name = "Ion 1"
				)
			},
			{
				"x": 'turn',
				"y": 'spill:ion2:accumulated',
				"settings": dict(
					mode = "lines",
					line = dict(
						color = "red"
					),
					name = "Ion 2"
				)
			}
		],
		plot_layout = spill_layout,
		category = "Turn By Turn"
	)

def process_particles_file(dashboard: ExtractionDashboard, particles: xt.Particles) -> dict:
	"""
	Maps the data needed extracted from the file according to `dashboard.data_to_expect`
	"""
	pass

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