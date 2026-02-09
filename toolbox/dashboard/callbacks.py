from __future__ import annotations
from dash import Input, Output, State, MATCH, no_update, Dash
from dash.exceptions import PreventUpdate
import os
import traceback
from pathlib import Path
import numpy as np
import numbers
import pandas as pd
import datetime
from toolbox.dashboard.profiles.datafield import Ratio


def _bin_array(arr, bin_length: int, how: str) -> list:
	if not arr: return []
	
	bins_length = (len(arr) // bin_length) * bin_length
	if bins_length == 0: return []
	
	if how == "first":
		return arr[:bins_length:bin_length]
	if how == "last":
		return arr[bin_length - 1:bins_length:bin_length]
	if how == "middle":
		return arr[bin_length // 2:bins_length:bin_length]
	
	if isinstance(arr[0], numbers.Real):
		a = np.asarray(arr[:bins_length])
		a_bined = a.astype(float, copy = False).reshape(-1, bin_length)
		
		if how == "sum":
			return a_bined.sum(axis = 1).tolist()
		if how == "mean":
			return a_bined.mean(axis = 1).tolist()	
		if how == "max":
			return a_bined.max(axis = 1).tolist()
		if how == "min":
			return a_bined.min(axis = 1).tolist()
		
		raise ValueError(how)
	elif isinstance(arr[0], Ratio):
		out = []
		for i in range(0, bins_length, bin_length):
			chunk = arr[i: i + bin_length]

			if how in {"sum", "mean"}:
				res = sum(chunk, Ratio(0, 0))
				out.append(res.value())
			elif how == "max":
				out.append(max(x.value() for x in chunk))
			elif how == "min":
				out.append(min(x.value() for x in chunk))
			else:
				raise ValueError(how)
		return out
	else:
		raise TypeError(f"Unsupported data type: {type(arr[0])}")

def register_callbacks(app: Dash, dashboard: ExtractionDashboard):
	
	#--General callbacks--
	@app.callback(
		Output("refresh", "disabled"),
		Input("mode-switch", "value"),
	)
	def toggle_interval(mode: str | None) -> bool:
		"""
		Toggles on/off the page update depending on the mode selected.
		For `mode` in `["file", None]` - swtiches off, for `live` - switches on.

		Parameters
		----------
		mode
			Either `live`, `file`, or None
		"""
		stop_update = (mode == "file") or (mode is None)
		return stop_update

	#--Live streaming callbacks--
	@app.callback(
		Output("listener-trigger", "children"),
		Input("mode-switch", "value"),
	)
	def _trigger_listener(mode: str | None):
		"""
		Switches on the listener daemon when the mode is `live`.
		In other cases, stops it.

		Parameters
		----------
		mode
			Either `live`, `file`, or None
		"""
		dashboard._clear_buffer()

		if mode is None:
			raise PreventUpdate

		if mode == "live":
			if not getattr(dashboard, "_listener_thread", None):
				dashboard.start_listener()
		else:
			if getattr(dashboard, "_listener_thread", None):
				dashboard.stop_listener()
				print(f"[INFO] Listener terminated.")

				del dashboard._listener_thread
		return ""

	@app.callback(
		Output({"type": "stream-graph", "key": MATCH}, "extendData"),
		Input("refresh", "n_intervals"),
		State("mode-switch", "value"),
		State({"type": "stream-graph", "key": MATCH}, "id"),
		State("bin-length", "value"),
	)
	def stream_data(_, mode: str | None, graph_id: dict, bin_length: int) -> tuple[dict, list, int]:
		"""
		Streams the data to a graph via `extendData`. Like that one does not have to rerender 
		the figure evey time the new data arrives.

		Parameters
		----------
		mode
			Either `live`, `file`, or None
		graph_id
			`dict` with a `dcc.Graph` id in a form `{"type": "stream-graph", "key": MATCH}`.
			With `MATCH` being unique id of a figure (Eg. `'intensity'`)
		bin_length
			Length of turns in 1 bin
		"""
		if mode in {"file", None}:
			return no_update

		data_key = graph_id["key"]

		df = dashboard.data_fields[data_key]
		bin_info = df.bin

		trace_bufs = df.plot_from or df.buffer_dependance

		trace_indices = []
		xs, ys = [], []

		with dashboard._buflock:
			total = len(dashboard.data_buffer[trace_bufs[0]].data)
			
			if not bin_info or not bin_info['enabled'] or bin_length <= 1:
				ptr = df.buffer_pointer
				if ptr >= total:
					return no_update
				
				end = min(ptr + dashboard.CHUNK_SIZE, total)

				for i, tmp in enumerate(df.plot_order):
					raw_x = dashboard.data_buffer[tmp['x']].data[ptr:end]

					if raw_x and isinstance(raw_x[0], datetime.datetime):
						x_vals = [dt.isoformat() for dt in raw_x]
					else:
						x_vals = raw_x

					try:
						y_vals = [float(y) for y in dashboard.data_buffer[tmp['y']].data[ptr:end]] 
					except TypeError:
						y_vals = [y.value() for y in dashboard.data_buffer[tmp['y']].data[ptr:end]] 
					
					xs.append(x_vals)
					ys.append(y_vals)

					trace_indices.append(i)

				df.buffer_pointer = end

				return dict(x = xs, y = ys), trace_indices, total

			# binned streaming
			else:
				ptr = getattr(df, "buffer_pointer_bin", 0)
				total = total // bin_length

				if ptr >= total:
					return no_update

				end = min(ptr + dashboard.CHUNK_SIZE, total)

				raw_start = ptr * bin_length
				raw_end = end * bin_length

				how_x = bin_info['x']
				how_y = bin_info['y']

				for i, tmp in enumerate(df.plot_order):
					raw_x = dashboard.data_buffer[tmp['x']].data[raw_start:raw_end]
					x_vals = _bin_array(raw_x, bin_length, how_x)

					if x_vals and isinstance(x_vals[0], datetime.datetime):
						x_vals = [dt.isoformat() for dt in x_vals]

					raw_y =  dashboard.data_buffer[tmp['y']].data[raw_start:raw_end]
					y_vals = _bin_array(raw_y, bin_length, how_y)
					
					xs.append(x_vals)
					ys.append(y_vals)

					trace_indices.append(i)

				df.buffer_pointer_bin = end

				return dict(x = xs, y = ys), trace_indices, total

	#--Reading from file callbacks--
	@app.callback(
		Output("file-suggest", "options"),
		Input("file-path", "value"),
	)
	def suggest_paths(text):
		if not text:
			return []

		expanded = os.path.expanduser(text)
		if expanded.endswith(os.sep) or Path(expanded).is_dir():
			dirpath = Path(expanded)
			prefix = ""
		else:
			p = Path(expanded)
			dirpath = p.parent if str(p.parent) else Path(".")
			prefix = p.name

		try:
			if not dirpath.exists() or not dirpath.is_dir():
				return []

			pref = prefix.lower()
			entries = []
			for child in dirpath.iterdir():
				name = child.name
				if prefix and not name.lower().startswith(pref):
					continue

				label = name + ("/" if child.is_dir() else "")
				value = str(child) + ("/" if child.is_dir() else "")
				entries.append({"label": label, "value": value})

			entries.sort(key = lambda d: (not d["label"].endswith("/"), d["label"].lower()))
			return entries[:50]

		except PermissionError:
			return [{"label": "Permission denied", "value": ""}]


	@app.callback(
		Output({"type":"stream-graph", "key": MATCH}, "figure"),
		Input("mode-switch", "value"),
		Input("bin-length", "value"),
		Input("cycle-selector", "value"),
		Input("cycle-load-trigger", "children"),
		State({"type":"stream-graph", "key": MATCH}, "id"),
		prevent_initial_call = True,
	)
	def render_full_figure(mode, bin_length, cycle_value, _loaded_trigger, graph_id):
		if mode != "file":
			return no_update
		
		data_key = graph_id["key"]
		df = dashboard.data_fields[data_key]
		bin_info = df.bin

		buff = {}
		with dashboard._buflock:
			for i, tmp in enumerate(df.plot_order):
				raw_x = dashboard.data_buffer[tmp["x"]].data
				raw_y = dashboard.data_buffer[tmp["y"]].data

				do_bin = bin_info and bin_info.get("enabled") and (bin_length and bin_length > 1)
				if do_bin:
					x = _bin_array(raw_x, bin_length, bin_info["x"])
					y = _bin_array(raw_y, bin_length, bin_info["y"])
				else:
					x = raw_x
					try:
						y = [float(v) for v in raw_y]
					except TypeError:
						y = [v.value() for v in raw_y]

				if tmp['x'] not in buff:
					buff[tmp['x']] = x
				if tmp['y'] not in buff:
					buff[tmp['y']] = y
		
		return dashboard.plot_figure(data_key, **buff)

	@app.callback(
		Output("file-path", "value"),
		Input("file-suggest", "value"),
		prevent_initial_call = True,
	)
	def apply_suggestion(selected):
		if not selected:
			return no_update
		return selected

	@app.callback(
		Output("file-controls", "style"),
		Input("mode-switch", "value")
	)
	def toggle_file_controls(mode):
		return {"display": "block"} if mode in ["file"] else {"display": "none"}

	@app.callback(
		Output("cycle-selector", "options"),
		Output("cycle-selector", "value"),
		Output("load-status", "children"),
		Input("load-file-btn", "n_clicks"),
		State("file-path", "value"),
		State("mode-switch", "value"),
		prevent_initial_call = True
	)
	def load_file_and_populate_cycles(n_clicks, filepath, mode):
		if not filepath or not Path(filepath).is_file():
			return [], None, "No file selected or not found."

		if mode == 'file':
			dashboard.data_from_file = dashboard.profile.read_file(filepath)

			if isinstance(dashboard.data_from_file, pd.DataFrame) and ("cycle_id" in dashboard.data_from_file.columns): 
				unique_cycles = sorted(getattr(dashboard.data_from_file, 'cycle_id').unique())
				options = [{"label": str(c), "value": c} for c in unique_cycles]
				default = unique_cycles[0] if unique_cycles else None
				return options, default, f"Loaded {len(unique_cycles)} cycles."
			else:
				return [0], 0, f"Loaded data."			
		else:
			return [], None, "Unknown error"

	@app.callback(
		Output("cycle-load-trigger", "children"),
		Input("mode-switch", "value"),
		Input("cycle-selector", "value"),
		State("file-path", "value"),
		prevent_initial_call = True
	)
	def on_cycle_selected(mode, cycle_id, filepath):
		if cycle_id is None:
			return no_update

		try:
			if mode == "live":
				return no_update

			elif mode == "file":
				dashboard._clear_buffer()

				data_mapping = dashboard.profile.process_file(dashboard, dashboard.data_from_file, cycle_id = cycle_id)

				with dashboard._buflock:
					dashboard.current_batch_id = 0
					for key in data_mapping:
						dashboard.data_buffer[key].extend(data_mapping[key], batch_id = dashboard.current_batch_id)
					
					dashboard.run_callbacks()

			print(f"[INFO] Loaded cycle #{cycle_id}")

		except Exception as e:
			print(f"[ERROR] reading cycle {cycle_id}: {e}")
			traceback.print_exc()
			return no_update