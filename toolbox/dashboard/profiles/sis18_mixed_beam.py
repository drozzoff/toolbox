from __future__ import annotations
import numpy as np
from functools import partial
import xtrack as xt
import pickle as pk
from toolbox.dashboard.profiles.datafield import DataField
from toolbox.dashboard.profiles.sis18 import SIS18Profile


class SIS18_mixed_beam_Profile:
	def __init__(self, *, ion1_chi: float, ion2_chi: float):
		self.ion1_chi = ion1_chi
		self.ion2_chi = ion2_chi

		self.base_profile = SIS18Profile()

	name = "SIS18 KO mixed beam"
	
	def make_datafields(self, dashboard: ExtractionDashboard):
		# reading the basic 
		res = self.base_profile.make_datafields(dashboard)
		
		# and extending for mixed ions beam
		res['spill:ion1'] =  DataField(
			buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn', 'extracted_at_ES:ion'], 
			output_buffers = ['spill:ion1'],
			callback = partial(ion_spill_callback, dashboard, 1, start_count_at_turn = 500),
			callback_level = 0,
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
			bin = dict(enabled = True, x = "middle", y = "sum"),
			plot_layout = partial(spill_layout, title = "Spill, Ion 1"),
			category = "Turn By Turn"
		)
		res['spill:ion2'] =  DataField(
			buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn', 'extracted_at_ES:ion'], 
			output_buffers = ['spill:ion2'],
			callback = partial(ion_spill_callback, dashboard, 2, start_count_at_turn = 500),
			callback_level = 0,
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
			bin = dict(enabled = True, x = "middle", y = "sum"),
			plot_layout = partial(spill_layout, title = "Spill, Ion 2"),
			category = "Turn By Turn"
		)
		res['spill:mixed'] =  DataField(
			buffer_dependance = ['turn', 'spill:ion1', 'spill:ion2'], 
			callback = partial(mixed_spill_callback, dashboard, start_count_at_turn = 500),
			callback_level = 1,
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
						name = "Ion 1",
						showlegend = True
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
						name = "Ion 2",
						showlegend = True
					)
				}
			],
			bin = dict(enabled = True, x = "middle", y = "sum"),
			plot_layout = partial(spill_layout, title = "Spill, mixed"),
			category = "Turn By Turn"
		)
		res['spill:ion1:accumulated'] =  DataField(
			buffer_dependance = ['turn', 'spill:ion1'], 
			output_buffers = ['spill:ion1:accumulated'],
			callback = partial(ion_accumulated_spill_callback, dashboard, 1),
			callback_level = 1,
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
			bin = dict(enabled = True, x = "last", y = "last"),
			plot_layout = partial(accumulated_spill_layout, title = "Accumulated spill, Ion 1"),
			category = "Turn By Turn"
		)
		res['spill:ion2:accumulated'] =  DataField(
			buffer_dependance = ['turn', 'spill:ion2'], 
			output_buffers = ['spill:ion2:accumulated'],
			callback = partial(ion_accumulated_spill_callback, dashboard, 2),
			callback_level = 1,
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
			bin = dict(enabled = True, x = "last", y = "last"),
			plot_layout = partial(accumulated_spill_layout, title = "Accumulated spill, Ion 2"),
			category = "Turn By Turn"
		)
		res['spill:mixed:accumulated'] =  DataField(
			buffer_dependance = ['turn', 'spill:ion1:accumulated', 'spill:ion2:accumulated'], 
			callback = partial(mixed_accumulated_spill_callback, dashboard),
			callback_level = 2,
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
						name = "Ion 1",
						showlegend = True
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
						name = "Ion 2",
						showlegend = True
					)
				}
			],
			bin = dict(enabled = True, x = "last", y = "last"),
			plot_layout = partial(accumulated_spill_layout, title = "Accumulated spill, mixed"),
			category = "Turn By Turn"
		)
		return res

	def read_file(self, filename: str) -> xt.Particles:
		return self.base_profile.read_file(filename)

	def process_file(self, dashboard: ExtractionDashboard, particles: xt.Particles | str, **kwargs) -> dict:
		"""
		Maps the data needed extracted from the file according to `dashboard.data_to_expect`
		"""
		if isinstance(particles, str):
			particles = self.read_file(particles)

		data_map = self.base_profile.process_file(dashboard, particles)
		
		lost_mask = particles.state == 0
		at_start = np.abs(particles.s) < 1e-7
		
		if 'extracted_at_ES:ion' in dashboard.data_to_expect:
			extracted_at_ES = lost_mask & at_start
			if np.sum(extracted_at_ES) > 0:
				lost_particles_at_ES_septum = particles.filter(extracted_at_ES)
			else:
				lost_particles_at_ES_septum = None # means no particles are lost

		# looping again to include the ion data buffer
		for key in dashboard.data_to_expect:
			if key == 'extracted_at_ES:ion':
				ion_arr = np.zeros_like(lost_particles_at_ES_septum.chi, dtype = int)
				ion_arr[np.abs(lost_particles_at_ES_septum.chi - self.ion1_chi) < 1e-7] = 1
				ion_arr[np.abs(lost_particles_at_ES_septum.chi - self.ion2_chi) < 1e-7] = 2
				
				data_map[key] = ion_arr

		return data_map
	
# callbacks
def ion_spill_callback(dashboard: ExtractionDashboard, ion_key = 1, start_count_at_turn: int = 0):
	x = np.array(dashboard.data_buffer['extracted_at_ES:x'].recent_data)
	px = np.array(dashboard.data_buffer['extracted_at_ES:px'].recent_data)
	ion_id = np.array(dashboard.data_buffer['extracted_at_ES:ion'].recent_data)
	
	start_turn = dashboard.data_buffer['turn'].recent_data[0]
	end_turn = dashboard.data_buffer['turn'].recent_data[-1]
	at_turn = np.array(dashboard.data_buffer['extracted_at_ES:at_turn'].recent_data) - start_turn

	threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
	lost_inside_septum = x > threshold
	good_turns = at_turn > start_count_at_turn
	good_particles = ~lost_inside_septum & good_turns
	ion_survived_inside_septum = (good_particles) & (ion_id == ion_key)

	ion_survived_inside_septum_at_turn = at_turn[ion_survived_inside_septum]

	ion_losses_at_turn = np.bincount(
		ion_survived_inside_septum_at_turn,
		minlength = end_turn - start_turn
	)

	dashboard.data_buffer[f'spill:ion{ion_key}'].extend(ion_losses_at_turn, batch_id = dashboard.current_batch_id)

def mixed_spill_callback(dashboard: ExtractionDashboard, start_count_at_turn: int = 0):
	if dashboard.data_buffer['spill:ion1'].last_batch_id != dashboard.current_batch_id:
		ion_spill_callback(dashboard, 1, start_count_at_turn)
	
	if dashboard.data_buffer['spill:ion2'].last_batch_id != dashboard.current_batch_id:
		ion_spill_callback(dashboard, 2, start_count_at_turn)

def _accumulated_quantity(dashboard: ExtractionDashboard, buffer_key: str, **kwargs):
	"""
	takes `buffer_key` and pushes the data to `"{buffer_key}:accumulated"`
	"""
	acc_buffer_key = kwargs.get('new_buffer_name', f"{buffer_key}:accumulated")

	recent_value = dashboard.data_buffer[acc_buffer_key].data[-1] if dashboard.data_buffer[acc_buffer_key].data else 0

	data = np.array(dashboard.data_buffer[buffer_key].recent_data)
	extension = recent_value + np.cumsum(data)

	dashboard.data_buffer[acc_buffer_key].extend(extension, batch_id = dashboard.current_batch_id)

def ion_accumulated_spill_callback(dashboard: ExtractionDashboard, ion_key = 1, start_count_at_turn: int = 0):
	if dashboard.data_buffer[f'spill:ion{ion_key}'].last_batch_id != dashboard.current_batch_id:
		ion_spill_callback(dashboard, 1, start_count_at_turn)
	
	_accumulated_quantity(dashboard, f"spill:ion{ion_key}")

def mixed_accumulated_spill_callback(dashboard: ExtractionDashboard, start_count_at_turn: int = 0):
	if dashboard.data_buffer[f'spill:ion1:accumulated'].last_batch_id != dashboard.current_batch_id:
		ion_accumulated_spill_callback(dashboard, 1, start_count_at_turn)

	if dashboard.data_buffer[f'spill:ion2:accumulated'].last_batch_id != dashboard.current_batch_id:
		ion_accumulated_spill_callback(dashboard, 2, start_count_at_turn)

# plotting layouts
def spill_layout(fig: go.Figure, *, title: str = "Spill"):
	fig.update_layout(
		title = title,
		xaxis_title = 'turn',
		yaxis_title = 'Spill [a.u.]',
		height = 400,
		showlegend = True
	)

def accumulated_spill_layout(fig: go.Figure, *, title: str = "Accumulated spill"):
	fig.update_layout(
		title = title,
		xaxis_title = 'turn',
		yaxis_title = 'Spill [a.u.]',
		height = 700,
		showlegend = True
	)

