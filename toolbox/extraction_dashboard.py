import dash
from dash import dcc, html, no_update
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import threading
import socket
import json
import sys
import numpy as np

data_buffer = {
	'turn': [],
	'Nparticles': [],
	'ES_septum_loss_anode_inside': [],
	'ES_septum_loss_anode_outside': [],
	'ES_septum_loss_anode_total': [],
	'ES_septum_loss_cathode': [],
	'x_extracted_at_ES': [],
	'px_extracted_at_ES': [],
	'x_extracted_at_MS': [],
	'px_extracted_at_MS': [],
	'spill': []
}

separatrix = {
	'x_stable': [],
	'px_stable': [],
	'x_unstable': [],
	'px_unstable': []
}

phase_space_size_ES, phase_space_size_MS, intensity_size, ES_lose_size, spill_size = 0, 0, 0, 0, 0
first_time, read_separatrix, plotted_separatrix = True, False, False

def start_listener(host, port):
	def run():
		global phase_space_size_ES, phase_space_size_MS, intensity_size, ES_lose_size, spill_size
		global read_separatrix

		srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		try:
			srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
		except AttributeError:
			pass

		srv.bind((host, port))
		srv.listen(1)
		
		while True:
			conn, addr = srv.accept()
			print(f"[INFO] Connection from {addr}")
			for key in data_buffer:
				data_buffer[key].clear()
			phase_space_size_ES, phase_space_size_MS, intensity_size, ES_lose_size, spill_size = 0, 0, 0, 0, 0

			buffer = ''
			try:
				while True:
					chunk = conn.recv(4096).decode()
					if not chunk:
						break
					buffer += chunk
					while '\n' in buffer:
						line, buffer = buffer.split('\n', 1)

						incoming = json.loads(line)

						for key in {'turn', 'Nparticles', 'x_extracted_at_ES', 'px_extracted_at_ES', 'x_extracted_at_MS', 'px_extracted_at_MS', 'ES_septum_loss_anode_outside'}:
							if key in incoming:
								data_buffer[key].extend(incoming[key])

						# parsing the extracted coordinates to account for the losses inside of the septum 
						counter, entered_ES_septum = 0, 0
						if 'x_extracted_at_ES' in incoming:
							for x, px in zip(incoming['x_extracted_at_ES'], incoming['px_extracted_at_ES']):
								if x > -0.055 - (px + 7.4e-3)**2/(2 * 1.7857e-3):
									counter += 1
							entered_ES_septum = len(incoming['x_extracted_at_ES'])
						data_buffer['ES_septum_loss_anode_inside'].append(counter)
						data_buffer['spill'].append(entered_ES_septum - counter)

						# parsing the separatrix data
						if 'separatrix' in incoming and not read_separatrix:
							separatrix['x_stable'] = incoming['separatrix']['x_stable']
							separatrix['px_stable'] = incoming['separatrix']['px_stable']
							separatrix['x_unstable'] = incoming['separatrix']['x_unstable']
							separatrix['px_unstable'] = incoming['separatrix']['px_unstable']
							read_separatrix = True
							

						data_buffer['ES_septum_loss_anode_total'].append(counter + data_buffer['ES_septum_loss_anode_inside'][-1])
			except json.JSONDecodeError as e:
				print("[ERROR] Invalid JSON", e)
			finally:
				conn.close()
				print("[INFO] Client disconnected, back to listening")

	threading.Thread(target = run, daemon = True).start()

def run_dash_server(host = '127.0.0.1', port = 18080):

	start_listener(host, port)

	app = dash.Dash("Slow extraction w/ xsuite")

	intro_text = '''
	### SIS18 slow extraction dashboard.
	'''
	app.layout = html.Div([
		dcc.Markdown(children = intro_text),
		
		dcc.Tabs([
			dcc.Tab(label = "Turn dependant data", children = [
				html.Div([
					dcc.Graph(id = 'Intensity'),
				], style = {'display': 'flex', 'gap': '10px'}),
				html.Div([
					dcc.Graph(id = 'Losses at ES'),
				], style = {'display': 'flex', 'gap': '10px'}),
				html.Div([
					dcc.Graph(id = 'Spill'),
				], style = {'display': 'flex', 'gap': '10px'})
			]),

				

			dcc.Tab(label = "Phase space", children = [
				html.Div([
					dcc.Graph(id = 'Phase space ES', style = {'flex': 1}),
					dcc.Graph(id = 'Phase space MS', style = {'flex': 1}),
				], style = {'display': 'flex', 'gap': '10px'}),
			]),
			dcc.Tab(label = "Separatrix", children = [
				html.Div([
					dcc.Graph(id = 'Separatrix', style = {'flex': 1}),
				], style = {'display': 'flex', 'gap': '10px'}),
			]),
		]),

		dcc.Interval(id = 'refresh', interval = 500, n_intervals = 0)
	])

	@app.callback(
		[
			Output('Intensity', 'figure'), 
			Output('Losses at ES', 'figure'),
			Output('Spill', 'figure'),
			Output('Phase space ES', 'figure'), 
			Output('Phase space MS', 'figure'),
			Output('Separatrix', 'figure')
		],
		[Input('refresh', 'n_intervals')]
	)

	def update_graph(n):
		global phase_space_size_ES, phase_space_size_MS, intensity_size, ES_lose_size, spill_size
		global first_time, read_separatrix, plotted_separatrix

		updates = [no_update, no_update, no_update, no_update, no_update, no_update]

		if intensity_size < len(data_buffer['Nparticles']) or first_time:
			intensity_fig = go.Figure(
				data = go.Scatter(
					x = data_buffer['turn'],
					y = data_buffer['Nparticles'], 
					mode = 'lines'
				)
			)
			intensity_fig.update_layout(
				title = 'Intensity',
				xaxis_title = 'Turn',
				yaxis_title = 'Number of particles in the ring',
				width = 1800,
				height = 400,
			)
			updates[0] = intensity_fig
			intensity_size = len(data_buffer['Nparticles'])

		if ES_lose_size < len(data_buffer['ES_septum_loss_anode_inside']) or first_time:
			trace3 = go.Scatter(
					x = data_buffer['turn'],
					y = data_buffer['ES_septum_loss_anode_inside'],
					mode = 'lines',
					line = {
						'color': 'blue',
						'width': 2,
					},
					name = "Lost inside of the septum",
					showlegend = True
				)

			trace2 = go.Scatter(
					x = data_buffer['turn'],
					y = data_buffer['ES_septum_loss_anode_outside'],
					mode = 'lines',
					line = {
						'color': 'red',
						'width': 2,
					},
					name = "Lost outside of the septum",
					showlegend = True
				)
			
			trace1 = go.Scatter(
					x = data_buffer['turn'],
					y = data_buffer['ES_septum_loss_anode_total'],
					mode = 'lines',
					line = {
						'color': 'green',
						'width': 2,
					},
					name = "Total losses the septum",
					showlegend = True
				)
			
			es_loss_fig = go.Figure(data = [trace1, trace2, trace3])
			
			es_loss_fig.update_layout(
				title = 'Es losses on the anode',
				xaxis_title = 'Turn',
				yaxis_title = 'Number of lost particles',
				width = 1800,
				height = 400,
			)
			updates[1] = es_loss_fig
			ES_lose_size = len(data_buffer['Nparticles'])

		if spill_size < len(data_buffer['spill']) or first_time:
			intensity_fig = go.Figure(
				data = go.Scatter(
					x = data_buffer['turn'],
					y = data_buffer['spill'], 
					mode = 'lines'
				)
			)
			intensity_fig.update_layout(
				title = 'Spill',
				xaxis_title = 'Turn',
				yaxis_title = 'Number of particles in the ring',
				width = 1800,
				height = 400,
			)
			updates[2] = intensity_fig
			spill_size = len(data_buffer['spill'])
		
		if phase_space_size_ES < len(data_buffer['x_extracted_at_ES']) or first_time:
			phase_space_fig = go.Figure(
				data = go.Scatter(
					x = data_buffer['x_extracted_at_ES'], 
					y = data_buffer['px_extracted_at_ES'], 
					mode = 'markers',
					marker = dict(
						size = 5,
						color = 'black',
						opacity = 0.3
					),
					name = "Extracted particles"
				)
			)

			# Anode
			phase_space_fig.add_shape(
				type = 'line',
				x0 = -0.055, y0 = -0.0085,
				x1 = -0.055, y1 = -0.005,
				line = dict(
					color = "LightSeaGreen",
					width = 4,
					dash = "dashdot",
				),
				name = "Anode",
				showlegend = True
			)

			# Cathode
			cathode_x_line = [-0.073, -0.073, -0.083, -0.083]
			cathode_y_line = [-0.0085, -0.005, -0.005, -0.0085]

			phase_space_fig.add_trace(go.Scatter(
				x = cathode_x_line,
				y = cathode_y_line,
				fill = 'toself',
				fillcolor = 'rgba(0, 0, 255, 0.3)',
				line = dict(color = 'rgba(0, 0, 0, 0)'),
				name = "Cathode",
			))

			# limits on ont being lost inside of the septum
			
			px_loss_limit = np.linspace(-7.4e-3, -5.0e-3, 100).tolist()
			x_loss_limit = list(map(lambda px: -0.055 - (px + 7.4e-3)**2 / (2 * 1.7857e-3), px_loss_limit))

			phase_space_fig.add_trace(go.Scatter(
				x = x_loss_limit,
				y = px_loss_limit,
				mode = 'lines',
				line = dict(
					color = 'red',
					dash = 'dash',
					),
				name = "Lost inside on the wires limit",
				showlegend = True
			))


			phase_space_fig.update_layout(
				title = 'Phase space at ES entrance',
				width = 800,
				height = 700,
				xaxis_title = 'x [m]',
				yaxis_title = 'px [rad]',
				showlegend = True
			)
			updates[3] = phase_space_fig

			phase_space_size_ES = len(data_buffer['x_extracted_at_ES'])

		if phase_space_size_MS < len(data_buffer['x_extracted_at_MS']) or first_time:
			phase_space_fig = go.Figure(
				data = go.Scatter(
					x = data_buffer['x_extracted_at_MS'], 
					y = data_buffer['px_extracted_at_MS'], 
					mode = 'markers',
					marker = dict(
						size = 5,
						color = 'black',
						opacity = 0.3
					),
					name = "Extracted particles"
				)
			)

			phase_space_fig.add_shape(
				type = 'line',
				x0 = 0.038, y0 = 0.003,
				x1 = 0.038, y1 = 0.009,
				line = dict(
					color = "LightSeaGreen",
					width = 4,
					dash = "dashdot",
				),
				name = "Bottom of the septum",
				showlegend = True
			)

			phase_space_fig.add_shape(
				type = 'line',
				x0 = 0.1, y0 = 0.003,
				x1 = 0.1, y1 = 0.009,
				line = dict(
					color = "SeaGreen",
					width = 4,
					dash = "dash",
				),
				name = "Top of the septum",
				showlegend = True
			)

			phase_space_fig.add_trace(go.Scatter(
				x = [0.069],
				y = [8e-3],
				mode = 'markers',
				name = 'Septum centeer orbit',
				marker = dict(
					symbol = 'x',
					size = 12,
					color = 'red',
					line = dict(width = 2)
				),
				showlegend = True
			))

			phase_space_fig.update_layout(
				title = 'Phase space at MS entrance',
				width = 800,
				height = 700,
				xaxis_title = 'x [m]',
				yaxis_title = 'px [rad]',
				showlegend = True
			)
			updates[4] = phase_space_fig

			phase_space_size_MS = len(data_buffer['x_extracted_at_MS'])

		if read_separatrix and not plotted_separatrix:
			trace1 = go.Scatter(
				x = separatrix['x_stable'],
				y = separatrix['px_stable'],
				mode = 'markers',
				marker = dict(
					size = 5,
					color = 'red',
				),
				name = "Unstable particle",
				showlegend = True
			)

			trace2 = go.Scatter(
				x = separatrix['x_unstable'],
				y = separatrix['px_unstable'],
				mode = 'markers',
				marker = dict(
					size = 5,
					color = 'green',
				),
				name = "Stable particle",
				showlegend = True
			)
			
			sep_fig = go.Figure(data = [trace1, trace2])

			sep_fig.add_shape(
				type = 'line',
				x0 = -0.055, y0 = -0.0085,
				x1 = -0.055, y1 = 0.005,
				line = dict(
					color = "SeaGreen",
					width = 4,
					dash = "dash",
				),
				name = "ES Septum anode",
				showlegend = True
			)

			sep_fig.update_layout(
				title = 'Separatrix',
				width = 1000,
				height = 800,
				xaxis_title = 'x [m]',
				yaxis_title = 'px [rad]',
				showlegend = True
			)

			updates[5] = sep_fig
			plotted_separatrix = True

		if first_time:
			first_time = False
		return updates

	
	try:
		print("[INFO] Starting Dash server...")
		app.run(debug = True, use_reloader = False)
	except KeyboardInterrupt:
		print("\n[INFO] Caught Ctrl+C. Cleaning up...")
		sys.exit(0)