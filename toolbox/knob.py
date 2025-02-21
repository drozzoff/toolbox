from pandas import DataFrame
from numpy import ndarray
import json
from xtrack import Line

class Knob:
	"""
	An object that I prefer more to the typical way, the knobs are implemented in `xsuite`.
	"""
	def __init__(self, input_data):
		self.modifications = None
		
		self.line = None

		if isinstance(input_data, str):
			with open(input_data) as f:
				self.modifications = json.load(f)
		elif isinstance(input_data, dict):
			self.modifications = input_data
		else:
			raise ValueError("Invalid input data")

	def is_attached(self):
		return not self.line is None

	def _check_compatibility(self):
		"""Check if the elements of the Knob align with the beamline"""

		if not self.is_attached():
			raise ValueError("Knob is not attached to any beamline")

		for element in self.modifications:
			if not element in self.line.element_names:
				raise ValueError(f"Element {element} not found in the line")

			for prop in self.modifications[element]:
				if not hasattr(self.line[element], prop):
					raise ValueError(f"Property {prop} not found in the element {element}")

	def attach_to(self, line: Line):
		"""
		Attach the `Knob` to a beamline so the knob can modify the beamline properties
		"""
		self.line = line

		self._check_compatibility()

	def apply(self, amplitude: float):
		"""
		Apply the knob
		"""

		if not self.is_attached():
			raise ValueError("Knob is not attached to any beamline")

		for element in self.modifications:
			for prop in self.modifications[element]:
				value = getattr(self.line[element], prop)
				modification = self.modifications[element][prop]

				# float/int, setting directly
				if isinstance(value, float) or isinstance(value, int):
					value += modification * amplitude
				
				# dict
				if isinstance(value, list) or isinstance(value, ndarray):
					# modification must be the dict than
					for key in modification:
						value[int(key)] += modification[key] * amplitude

				# if the attribute is mutable this is not needed
				setattr(self.line[element], prop, value)

	def to_dataframe(self) -> DataFrame:
		"""Get the DataFrame representation"""
		columns = ['name']

		for element in self.modifications:
			for prop in self.modifications[element]:
				if not prop in columns:
					columns.append(prop)

		data_dict = {key: [] for key in columns}

		for element in self.modifications:
			data_dict['name'].append(element)

			for prop in columns:
				if prop == 'name':
					continue

				if prop in self.modifications[element]:
					data_dict[prop].append(self.modifications[element][prop])
				else:
					data_dict[prop].append(None)
				

		return DataFrame(data_dict)

	def __str__(self):
		
		res = f"Knob object\nis_attached = {self.is_attached()}\nline = {str(self.line)}\n"
		res += str(self.to_dataframe())
		return res

	__repr__ = __str__