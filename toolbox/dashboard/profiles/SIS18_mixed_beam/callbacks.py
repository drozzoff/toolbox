from __future__ import annotations
import numpy as np


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


"""
	ASSORTED CALLBACKS	

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


"""