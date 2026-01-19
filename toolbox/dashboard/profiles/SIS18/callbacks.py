from __future__ import annotations
import numpy as np


def ES_inside_losses_callback(dashboard: ExtractionDashboard):

	x = np.array(dashboard.data_buffer['extracted_at_ES:x'].recent_data)
	px = np.array(dashboard.data_buffer['extracted_at_ES:px'].recent_data)

	start_turn = dashboard.data_buffer['turn'].recent_data[0]
	end_turn = dashboard.data_buffer['turn'].recent_data[-1]

	at_turn = np.array(dashboard.data_buffer['extracted_at_ES:at_turn'].recent_data) - start_turn

	threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
	lost_inside_septum_at_turn = at_turn[x > threshold]

	losses_at_turn = np.bincount(
		lost_inside_septum_at_turn,
		minlength = end_turn - start_turn
	)
	dashboard.data_buffer['ES_septum_losses:inside'].extend(losses_at_turn, batch_id = dashboard.current_batch_id)

def ES_losses_callback(dashboard: ExtractionDashboard):
	if dashboard.data_buffer['ES_septum_losses:inside'].last_batch_id != dashboard.current_batch_id:
		ES_inside_losses_callback(dashboard)

	lost_inside = np.array(dashboard.data_buffer['ES_septum_losses:inside'].recent_data)
	lost_outside = np.array(dashboard.data_buffer['lost_on_septum_wires'].recent_data)

	dashboard.data_buffer['ES_septum_losses'].extend(lost_inside + lost_outside, batch_id = dashboard.current_batch_id)

def spill_callback(dashboard: ExtractionDashboard):
	x = np.array(dashboard.data_buffer['extracted_at_ES:x'].recent_data)
	px = np.array(dashboard.data_buffer['extracted_at_ES:px'].recent_data)

	start_turn = dashboard.data_buffer['turn'].recent_data[0]
	end_turn = dashboard.data_buffer['turn'].recent_data[-1]

	at_turn = np.array(dashboard.data_buffer['extracted_at_ES:at_turn'].recent_data) - start_turn

	threshold = -0.055 - (px + 7.4e-3)**2  / (2 * 1.7857e-3)
	lost_inside_septum = x > threshold
	survived_inside_septum_at_turn = at_turn[~lost_inside_septum]

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