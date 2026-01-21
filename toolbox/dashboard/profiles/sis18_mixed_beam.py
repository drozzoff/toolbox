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

	name = "SIS18_mixed_beam"
	
	def make_datafields(self, dashboard: ExtractionDashboard):
		# reading the basic 
		res = self.base_profile.make_datafields(dashboard)
		
		# and extending for mixed ions beam
		res['spill:ion1'] =  DataField(
			buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn', 'extracted_at_ES:ion'], 
			output_buffers = ['spill:ion1'],
			callback = partial(ion_spill_callback, dashboard, 1),
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
			plot_layout = spill_layout,
			category = "Turn By Turn"
		)
		res['spill:ion2'] =  DataField(
			buffer_dependance = ['turn', 'extracted_at_ES:x', 'extracted_at_ES:px', 'extracted_at_ES:at_turn', 'extracted_at_ES:ion'], 
			output_buffers = ['spill:ion2'],
			callback = partial(ion_spill_callback, dashboard, 2),
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
			plot_layout = spill_layout,
			category = "Turn By Turn"
		)
		res['spill:mixed'] =  DataField(
			buffer_dependance = ['turn', 'spill:ion1', 'spill:ion2'], 
			callback = partial(mixed_spill_callback, dashboard),
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
			plot_layout = accumulated_spill_layout,
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
			plot_layout = accumulated_spill_layout,
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
			plot_layout = accumulated_spill_layout,
			category = "Turn By Turn"
		)
		return res

	def process_file(self, dashboard: ExtractionDashboard, particles: xt.Particles | str, **kwargs) -> dict:
		"""
		Maps the data needed extracted from the file according to `dashboard.data_to_expect`
		"""
		if isinstance(particles, str):
			with open(particles, 'rb') as fid:
				particles = xt.Particles.from_dict(pk.load(fid))

			particles.sort(by = 'at_turn', interleave_lost_particles = True)

		data_map = self.base_profile.process_file(dashboard, particles)
		print(particles.get_table())
		chi = particles.chi
		print(np.sum(chi == 1.0))
		# looping again to include the ion data buffer
		for key in dashboard.data_to_expect:
			if key == 'extracted_at_ES:ion':
				ion_arr = np.zeros_like(particles.chi, dtype = int)
				ion_arr[np.abs(particles.chi - self.ion1_chi) < 1e-7] = 1
				ion_arr[np.abs(particles.chi - self.ion2_chi) < 1e-7] = 2
				
				data_map[key] = ion_arr

		return data_map
	
# callbacks
def ion_spill_callback(dashboard: ExtractionDashboard, ion_key = 1):
	x = np.array(dashboard.data_buffer['extracted_at_ES:x'].recent_data)
	px = np.array(dashboard.data_buffer['extracted_at_ES:px'].recent_data)

	start_turn = dashboard.data_buffer['turn'].recent_data[0]
	end_turn = dashboard.data_buffer['turn'].recent_data[-1]

	at_turn = np.array(dashboard.data_buffer['extracted_at_ES:at_turn'].recent_data) - start_turn

	threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
	lost_inside_septum = x > threshold

	survived_inside_septum_at_turn = at_turn[~lost_inside_septum]

	ion_id = np.array(dashboard.data_buffer['extracted_at_ES:ion'].recent_data)

	print(np.sum(ion_id == 1))
	print(np.sum(ion_id == 2))
	ion_survived_inside_septum_at_turn = survived_inside_septum_at_turn[ion_id == ion_key]

	ion_losses_at_turn = np.bincount(
		ion_survived_inside_septum_at_turn,
		minlength = end_turn - start_turn
	)

	dashboard.data_buffer[f'spill:ion{ion_key}'].extend(ion_losses_at_turn, batch_id = dashboard.current_batch_id)

def mixed_spill_callback(dashboard: ExtractionDashboard):
	if dashboard.data_buffer['spill:ion1'].last_batch_id != dashboard.current_batch_id:
		ion_spill_callback(dashboard, 1)
	
	if dashboard.data_buffer['spill:ion2'].last_batch_id != dashboard.current_batch_id:
		ion_spill_callback(dashboard, 2)

def _accumulated_quantity(dashboard: ExtractionDashboard, buffer_key: str, **kwargs):
	"""
	takes `buffer_key` and pushes the data to `"{buffer_key}:accumulated"`
	"""
	acc_buffer_key = kwargs.get('new_buffer_name', f"{buffer_key}:accumulated")

	recent_value = dashboard.data_buffer[acc_buffer_key].data[-1] if dashboard.data_buffer[acc_buffer_key].data else 0

	data = np.array(dashboard.data_buffer[buffer_key].recent_data)
	extension = recent_value + np.cumsum(data)

	dashboard.data_buffer[acc_buffer_key].extend(extension, batch_id = dashboard.current_batch_id)

def ion_accumulated_spill_callback(dashboard: ExtractionDashboard, ion_key = 1):
	if dashboard.data_buffer[f'spill:ion{ion_key}'].last_batch_id != dashboard.current_batch_id:
		ion_spill_callback(dashboard, 1)
	
	_accumulated_quantity(dashboard, 'spill:ion{ion_key}')

def mixed_accumulated_spill_callback(dashboard: ExtractionDashboard):
	if dashboard.data_buffer[f'spill:ion1:accumulated'].last_batch_id != dashboard.current_batch_id:
		ion_accumulated_spill_callback(dashboard, 1)

	if dashboard.data_buffer[f'spill:ion2:accumulated'].last_batch_id != dashboard.current_batch_id:
		ion_accumulated_spill_callback(dashboard, 2)

# plotting layouts
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

