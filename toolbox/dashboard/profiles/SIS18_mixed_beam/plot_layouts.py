from __future__ import annotations
import numpy as np

def spill_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Spill',
		xaxis_title = 'turn',
		yaxis_title = 'Spill [a.u.]',
		width = 2250,
		height = 400,
		showlegend = False
	)

def accumulated_spill_layout(fig: go.Figure):
	fig.update_layout(
		title = 'Spill accumulated',
		xaxis_title = 'turn',
		yaxis_title = 'Spill [a.u.]',
		width = 1200,
		height = 700,
		showlegend = False
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