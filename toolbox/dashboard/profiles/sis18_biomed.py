from __future__ import annotations
import pandas as pd
from toolbox.dashboard.profiles.datafield import DataField


class SIS18_biomed_Profile:
	def __init__(self):
		pass

	name = "SIS18biomed"
	def make_datafields(self, dashboard: ExtractionDashboard):
		return{
			'intensity': DataField(
				buffer_dependance = ['time', 'IC1', 'IC2', 'IC3'],
				plot_order = [
					{
						"x": 'time',
						"y": "IC3",
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "green",
							),
							name = "IC3"
						)
					},
					{
						"x": 'time',
						"y": "IC2",
						"settings": dict(
							mode = "lines",
							line = dict(
								color = "red",
							),
							name = "IC2"
						)
					},
					{
						"x": 'time',
						"y": "IC1",
						"settings": dict(
							mode = "lines",
							line = dict(
								color =  "blue",
							),
							name = "IC1"
						)
					},
				],
				plot_layout = biomed_data_layout,
				category = "Biomed"
			),
		}
	
	def read_file(self, filename: str) -> pd.DataFrame:
		return pd.read_parquet(filename)

	def process_file(self, dashboard: ExtractionDashboard, data: pd.DataFrame | str, **kwargs):
		if isinstance(data, str):
			data = self.read_file(data)

		cycle_id = kwargs.get("cycle_id", 0)
		single_cycle = data[data['cycle_id'] == cycle_id]
		print(single_cycle)

		data_mapping = {}
		for key in dashboard.data_to_expect:
			if key == 'time':
				data_mapping[key] = single_cycle.index.to_pydatetime()
			
			intensity_names = {'IC1': 'Y[0]', 'IC2': 'Y[1]', 'IC3': 'Y[2]'}
			if key in intensity_names:	
				data_mapping[key] = single_cycle[intensity_names[key]].values
			
		return data_mapping

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
