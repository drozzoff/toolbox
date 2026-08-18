from __future__ import annotations
import pandas as pd
from toolbox.dashboard.models import DataField, LoadedFile, FileSelection


class SIS18extraction_biomed:
	def __init__(self):
		pass

	name = "SIS18 slow extraction based on 3 ICs data"
	def make_datafields(self, dashboard: Dashboard):
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
	
	def make_infofields(self, dashboard: Dashboard):
		return {}

	def read_file(self, filename: str) -> LoadedFile:
		dataframe = pd.read_parquet(filename)

		cycles = sorted(dataframe["cycle_id"].unique())

		return LoadedFile(
			data = dataframe,
			selections = [
				FileSelection(
					value = int(cycle),
					label = f"Cycle {cycle}"
				)
				for cycle in cycles
			],
			selection_name = "Cycle"
		)

	def process_file(
		self, 
		dashboard: Dashboard, 
		data: pd.DataFrame | str, 
		selection_id: int = 0 # Its the Cycle id in the dataframe
		):
		if isinstance(data, str):
			data = self.read_file(data)

		single_cycle = data[data['cycle_id'] == selection_id]
#		print(single_cycle)

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
		width = 1500,
		height = 800,
	)
