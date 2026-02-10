from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataField:
	"""Simple class to store the dependances and state of the data field in the dashboard"""
	buffer_dependance: list[str] = field(default_factory = list)
	output_buffers: list[str] | None = None
	callback: Optional[callable] = None
	callback_level: int | None = None
	state: bool = False
	plot_from: list[str] | None = None # list of the buffers that this data field is dependent upon for plotting
	bin: dict | None = None # Description of the binning to apply to the buffers when plotting
	plot_layout: Optional[callable] = None

	buffer_pointer: int = 0

	plot_order: list[dict] | None = None # Description of the order traces are added to the plot

	category: str | None = None

@dataclass
class InfoField:
	"""Simple class to store the info/summary of the buffers"""
	buffer_dependance: list[str] = field(default_factory = list)
	output_info: list[str] | str | None = None
	callback: Optional[callable] = None
	callback_level: int | None = None
#	render_message: 
	state: bool = False


@dataclass
class Ratio:
	num: int
	den: int

	def __add__(self, other: "Ratio") -> "Ratio":
		return Ratio(self.num + other.num, self.den + other.den)
	
	def value(self) -> float:
		return float(self.num) / float(self.den) if self.den != 0 else 0.0
	
	def __str__(self):
		return f"{self.num}/{self.den}"