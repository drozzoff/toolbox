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
	plot_layout: Optional[callable] = None

	buffer_pointer: int = 0

	plot_order: list[dict] | None = None # Description of the order traces are added to the plot

	category: str | None = None
	