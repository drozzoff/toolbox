from __future__ import annotations
from dash import html, dcc
from string import Formatter


def extract_keys(template: str) -> set[str]:
	formatter = Formatter()

	return {
		field_name
		for _, field_name, _, _ in formatter.parse(template)
		if field_name
	}

def make_layout(dashboard: ExtractionDashboard):

	intro_text = '''
	### SIS18 slow extraction dashboard.
	'''
	divs = {
		'turn_dependent_data': [],
		'phase_space': [],
		'separatrix': [],
		'special': []
	}
	info_divs = []
	for key in dashboard.data_to_monitor:
		if key in dashboard.data_fields:
			# Tab 1 - Turn dependent data
			if dashboard.data_fields[key].category == "Turn By Turn":
				divs['turn_dependent_data'].append(	
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = dashboard.plot_figure(key, init_run = True),
							style = {"width": "100%"},
							config = {"responsive": True}
						),
					], style = {'display': 'flex', "flexDirection":"column", 'gap': '10px'})
				)
			
			# Tab 2 - Phase space
			if dashboard.data_fields[key].category == "Phase Space":
				divs['phase_space'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = dashboard.plot_figure(key, init_run = True)
						)
					], style = {'display': 'flex', 'gap': '10px'})
				)
			
			# Tab 3 - Separatrix
			if dashboard.data_fields[key].category == "Separatrix":
				divs['separatrix'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = dashboard.plot_figure(key, init_run = True)
						)
					], style = {'display': 'flex', 'gap': '10px'}))

			if dashboard.data_fields[key].category == "Biomed":
				divs['special'].append(
					html.Div([
						dcc.Graph(
							id = {"type": "stream-graph","key": key},
							figure = dashboard.plot_figure(key, init_run = True)
						)
					], style = {'display': 'flex', 'gap': '10px'})
				)
		if key in dashboard.info_fields:
			field = dashboard.info_fields.get(key)
			info_divs.append(
				dcc.Markdown(
					id = {"type": "info-md", "key": key},
					children = field.template.format(**{k: dashboard.info_dict.get(k) for k in extract_keys(field.template)}),
					className =  "info-panel"
				)
			)
	tabs = []
	for key in divs:
		if divs[key] != []:
			tabs.append(dcc.Tab(label = key, children = divs[key]))

	layout = html.Div(
		className = "container",
		children = [
			html.Div(
				className = "topbar",
				children = [
					dcc.Markdown(
						children = intro_text,
						className = "intro"
					),
					html.Div(
						f"Profile: {dashboard.profile.name}",
						id = "profile-banner",
						className = "profile-badge",
					),
				],
			),

			html.Div(
				className = "main-grid",
				children = [
					html.Div(
						className = "left-panel",
						children = [

							# Controls card
							html.Div(
								className = "card",
								children = [
									html.Div(
										"Controls",
										className = "card-title",
									),

									html.Div(
										className = "control-row",
										children = [
											html.Span(
												"Mode",
												className = "label",
											),
											dcc.Dropdown(
												id="mode-switch",
												options=[
													{"label": "Live", "value": "live"},
													{"label": "From file", "value": "file"},
												],
												value = None,
												placeholder = "Select mode…",
												clearable = False,
												className = "dropdown small",
											),
										],
									),

									html.Div(
										className = "control-row",
										children = [
											html.Span(
												"Bin length",
												className = "label",
											),
											dcc.Dropdown(
												id = "bin-length",
												options = [
													{"label": str(x), "value": x}
													for x in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
												],
												value = 1,
												clearable = False,
												className = "dropdown small",
											),
										],
									),
								],
							),
							html.Pre(id = "client-debug", style = {"whiteSpace": "pre-wrap"}),
							html.Div(
								id = "file-controls",
								className = "card",
								style = {"display": "none"},
								children = [
									html.Div(
										"Load from file",
										className = "card-title",
									),
									dcc.Input(
										id = "file-path",
										type = "text",
										placeholder = "Type a path",
										debounce = False,
										className = "input",
									),
									dcc.Dropdown(
										id = "file-suggest",
										options = [],
										placeholder = "Suggestions",
										searchable = True,
										clearable = True,
										className = "dropdown",
									),
									html.Button(
										"Load file",
										id = "load-file-btn",
										className = "button-primary",
									),
									dcc.Dropdown(
										id = "cycle-selector",
										placeholder = "Select cycle",
										clearable = False,
										searchable = True,
										className = "dropdown small",
									),
									html.Div(
										id = "load-status",
										className = "status-text",
									),
									html.Div(
										id = "cycle-load-trigger",
										style = {"display": "none"},
									),
								],
							),
						],
					),
					html.Div(
						className = "plot-panel",
						children = [
							html.Div(
								className = "card main-card",
								children = [
									dcc.Tabs(
										tabs,
										className = "tabs",
									),
								],
							),
						],
					),
					html.Div(
						className = "right-panel",
						children = [
							html.Div(
									className = "card main-card",
									children = info_divs,
								),
						],
					),

				],
			),

			html.Div(
				id = "xaxis-trigger",
				style = {"display": "none"},
			),
			html.Div(
				id = "listener-trigger",
				style = {"display": "none"},
			),

			dcc.Interval(
				id = "refresh",
				interval = 200,
				n_intervals = 0,
			),
		],
	)

	return layout