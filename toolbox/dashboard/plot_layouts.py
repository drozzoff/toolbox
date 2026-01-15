import numpy as np
from __future__ import annotations

def intensity_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Intensity',
		xaxis_title = 'turn',
		yaxis_title = 'Intensity [a.u.]',
		width = 1800,
		height = 400,
	)

def ES_outside_losses_layout(fig: go.Figure):		
	fig.update_layout(
		title = 'Es losses on the outside of anode',
		xaxis_title = 'turn',
		yaxis_title = 'Lost particles [a.u.]',
		width = 1800,
		height = 400,
	)

def ES_inside_losses_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Es losses on the inside of anode',
		xaxis_title = 'turn',
		yaxis_title = 'Lost particles [a.u.]',
		width = 1800,
		height = 400,
	)

def ES_losses_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Es losses on the anode',
		xaxis_title = 'turn',
		yaxis_title = 'Lost [a.u.]',
		width = 1800,
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
		width = 2250,
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
		width = 1200,
		height = 700,
		showlegend = False
	)

def accumulated_ES_losses_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Accumulated losses on the anode',
		xaxis_title = 'turn',
		yaxis_title = 'Lost [a.u.]',
		width = 1500,
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

def biomed_data_layout(fig: go.Figure):
	fig.update_xaxes(
		type = "date",
		tickformat = "%H:%M:%S",
		tickangle = 0,
		showgrid = True,
	)
	
	fig.update_layout(
		title = 'Spill, biomed data',
		xaxis_title = 'time',
		yaxis_title = 'Spill',
		width = 2250,
		height = 900,
	)

"""

	ASSORTED DATA

match key:
	case 'ES_septum_anode_losses_mixed_accumulated':
		fig.update_layout(
			title = 'Accumulated losses on the anode',
			xaxis_title = self.time_coord,
			yaxis_title = 'Lost [a.u.]',
			width = 1500,
			height = 700,
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
"""